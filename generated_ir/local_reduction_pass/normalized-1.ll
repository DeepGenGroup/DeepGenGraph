source_filename = "/data2/xsl/DeepGenGraph/generated_ir/online_softmax_native_acc/final-export/run-20261023-182556-l6_z6lbl/input-0001-test_input_1x32x4096x128-128-128-32-online/retained32-baseline/input.mlir"

@shm_0 = addrspace(3) global [16384 x i8] undef, align 16

; Function Attrs: convergent mustprogress nofree willreturn memory(read)
; Unknown intrinsic
declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32) #0

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) nofree writeonly captures(none) %3) local_unnamed_addr #1 !reqd_work_group_size !1 {
  %5 = tail call i32 @llvm.amdgcn.workgroup.id.x()
  %6 = sext i32 %5 to i64
  %7 = tail call i32 @llvm.amdgcn.workgroup.id.y()
  %8 = sext i32 %7 to i64
  %9 = shl nsw i64 %8, 5
  %10 = add nsw i64 %9, %6
  %11 = lshr i64 %10, 5
  %12 = and i64 %11, 576460752303423484
  %13 = and i64 %6, 3
  %14 = or disjoint i64 %12, %13
  %15 = sub nsw i64 31, %14
  %16 = lshr i64 %10, 2
  %17 = icmp ugt i64 %10, 1023
  %18 = add nsw i64 %14, -32
  %19 = select i1 %17, i64 %18, i64 %15
  %20 = sdiv i64 %19, 32
  %21 = sext i1 %17 to i64
  %22 = xor i64 %20, %21
  %23 = add nsw i64 %22, %16
  %24 = shl i64 %15, 7
  %25 = shl i64 %22, 12
  %26 = sub i64 %24, %25
  %27 = shl i64 %23, 19
  %28 = and i64 %27, 16252928
  %29 = tail call i32 @llvm.amdgcn.workitem.id.x(), !range !2
  %30 = shl i64 %26, 8
  %31 = tail call i32 asm sideeffect "", "=v,0"(i32 %29) #9, !range !2, !frisk.register_lifetimes !3
  %32 = zext nneg i32 %31 to i64
  %33 = shl nuw nsw i64 %32, 3
  %34 = and i64 %33, 1536
  %35 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %34
  %36 = shl nuw nsw i64 %32, 7
  %37 = and i64 %36, 16256
  %38 = lshr i64 %32, 4
  %39 = and i64 %38, 8
  %40 = or disjoint i64 %37, %39
  %41 = ptrtoint ptr addrspace(1) %0 to i64
  %42 = or disjoint i64 %40, %28
  %43 = shl nuw nsw i64 %42, 1
  %44 = add i64 %43, %30
  %45 = trunc i64 %44 to i32
  %46 = insertelement <2 x i64> poison, i64 %41, i64 0
  %47 = bitcast <2 x i64> %46 to <4 x i32>
  %48 = shufflevector <4 x i32> %47, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  %49 = or disjoint i64 %32, 256
  %50 = shl nuw nsw i64 %49, 3
  %51 = and i64 %50, 3584
  %52 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %51
  %53 = lshr i64 %49, 4
  %54 = and i64 %53, 24
  %55 = or disjoint i64 %54, %37
  %56 = or disjoint i64 %55, %28
  %57 = shl nuw nsw i64 %56, 1
  %58 = add i64 %57, %30
  %59 = trunc i64 %58 to i32
  %60 = tail call i32 @llvm.amdgcn.workgroup.id.z()
  %61 = sext i32 %60 to i64
  %62 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %34
  %63 = shl nuw nsw i64 %28, 1
  %64 = add i64 %63, %30
  %65 = shl nuw nsw i64 %40, 1
  %66 = or disjoint i64 %65, %64
  %67 = trunc i64 %66 to i32
  %68 = or disjoint i32 %67, 64
  %69 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %51
  %70 = insertelement <4 x ptr addrspace(3)> poison, ptr addrspace(3) %35, i64 0
  %71 = insertelement <4 x ptr addrspace(3)> %70, ptr addrspace(3) %52, i64 1
  %72 = insertelement <4 x ptr addrspace(3)> %71, ptr addrspace(3) %62, i64 2
  %73 = insertelement <4 x ptr addrspace(3)> %72, ptr addrspace(3) %69, i64 3
  %74 = ptrtoint <4 x ptr addrspace(3)> %73 to <4 x i64>
  %75 = trunc <4 x i64> %74 to <4 x i32>
  %76 = extractelement <4 x i32> %75, i64 0
  %77 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %76)
  %78 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %77)
  %79 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %78, i32 %45, <4 x i32> %48) #10
  %80 = extractelement <4 x i32> %75, i64 1
  %81 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %80)
  %82 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %81)
  %83 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %82, i32 %59, <4 x i32> %48) #10
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %84 = extractelement <4 x i32> %75, i64 2
  %85 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %84)
  %86 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %85)
  %87 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %86, i32 %68, <4 x i32> %48) #10
  %88 = extractelement <4 x i32> %75, i64 3
  %89 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %88)
  %90 = shl nuw nsw i64 %55, 1
  %91 = or disjoint i64 %90, %64
  %92 = trunc i64 %91 to i32
  %93 = or disjoint i32 %92, 64
  %94 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %89)
  %95 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %94, i32 %93, <4 x i32> %48) #10
  %96 = lshr i32 %31, 2
  %97 = and i32 %96, 4
  %98 = shl nuw nsw i32 %31, 1
  %99 = and i32 %98, 384
  %100 = or disjoint i32 %97, %99
  %101 = shl nuw nsw i32 %31, 3
  %102 = and i32 %101, 120
  %103 = or disjoint i32 %100, %102
  %104 = shl nuw nsw i32 %31, 5
  %105 = and i32 %104, 1024
  %106 = or disjoint i32 %103, %105
  %107 = zext nneg i32 %106 to i64
  %108 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %107
  %109 = load <4 x half>, ptr addrspace(3) %108, align 8
  %110 = getelementptr i8, ptr addrspace(3) %108, i64 4096
  %111 = load <4 x half>, ptr addrspace(3) %110, align 8
  %112 = fmul <4 x half> %109, splat (half 1.275630e-01)
  %113 = getelementptr i8, ptr addrspace(3) %108, i64 1024
  %114 = load <4 x half>, ptr addrspace(3) %113, align 8
  %115 = fmul <4 x half> %111, splat (half 1.275630e-01)
  %116 = getelementptr i8, ptr addrspace(3) %108, i64 5120
  %117 = load <4 x half>, ptr addrspace(3) %116, align 8
  %118 = fmul <4 x half> %114, splat (half 1.275630e-01)
  %119 = fmul <4 x half> %117, splat (half 1.275630e-01)
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %120 = or disjoint i32 %67, 128
  %121 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %78, i32 %120, <4 x i32> %48) #10
  %122 = or disjoint i32 %92, 128
  %123 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %82, i32 %122, <4 x i32> %48) #10
  %124 = getelementptr i8, ptr addrspace(3) %108, i64 8192
  %125 = load <4 x half>, ptr addrspace(3) %124, align 8
  %126 = getelementptr i8, ptr addrspace(3) %108, i64 12288
  %127 = load <4 x half>, ptr addrspace(3) %126, align 8
  %128 = getelementptr i8, ptr addrspace(3) %108, i64 9216
  %129 = load <4 x half>, ptr addrspace(3) %128, align 8
  %130 = fmul <4 x half> %125, splat (half 1.275630e-01)
  %131 = getelementptr i8, ptr addrspace(3) %108, i64 13312
  %132 = load <4 x half>, ptr addrspace(3) %131, align 8
  %133 = fmul <4 x half> %127, splat (half 1.275630e-01)
  %134 = fmul <4 x half> %129, splat (half 1.275630e-01)
  %135 = fmul <4 x half> %132, splat (half 1.275630e-01)
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %136 = or disjoint i32 %67, 192
  %137 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %86, i32 %136, <4 x i32> %48) #10
  %138 = or disjoint i32 %92, 192
  %139 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %94, i32 %138, <4 x i32> %48) #10
  %140 = load <4 x half>, ptr addrspace(3) %108, align 8
  %141 = load <4 x half>, ptr addrspace(3) %110, align 8
  %142 = load <4 x half>, ptr addrspace(3) %113, align 8
  %143 = fmul <4 x half> %140, splat (half 1.275630e-01)
  %144 = load <4 x half>, ptr addrspace(3) %116, align 8
  %145 = fmul <4 x half> %141, splat (half 1.275630e-01)
  %146 = fmul <4 x half> %142, splat (half 1.275630e-01)
  %147 = fmul <4 x half> %144, splat (half 1.275630e-01)
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %148 = load <4 x half>, ptr addrspace(3) %124, align 8
  %149 = load <4 x half>, ptr addrspace(3) %126, align 8
  %150 = load <4 x half>, ptr addrspace(3) %128, align 8
  %151 = fmul <4 x half> %148, splat (half 1.275630e-01)
  %152 = load <4 x half>, ptr addrspace(3) %131, align 8
  %153 = fmul <4 x half> %149, splat (half 1.275630e-01)
  %154 = fmul <4 x half> %150, splat (half 1.275630e-01)
  %155 = fmul <4 x half> %152, splat (half 1.275630e-01)
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %156 = or disjoint i64 %24, 127
  %157 = add i64 %24, 128
  %158 = icmp uge i64 %157, %156
  %159 = sub nsw i64 32, %14
  %160 = shl i64 %16, 7
  %161 = and i64 %160, 3968
  %162 = shl nsw i64 %61, 12
  %163 = or disjoint i64 %161, %162
  %164 = shl nuw nsw i32 %29, 2
  %165 = and i32 %164, 12
  %166 = and i32 %29, 1
  %167 = or disjoint i32 %165, %166
  %168 = trunc nuw nsw i32 %167 to i16
  %169 = add nuw nsw i16 %168, 1
  %170 = lshr i16 %169, 1
  %171 = zext nneg i16 %170 to i64
  %172 = icmp ult i64 %10, 1024
  br i1 %172, label %180, label %173

173:                                              ; preds = %4
  %174 = tail call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %175 = tail call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %174)
  %176 = add nuw nsw i32 %175, 64
  %177 = xor i32 %175, 16
  %178 = and i32 %176, 192
  %179 = xor i32 %175, 32
  br label %1799

180:                                              ; preds = %4
  %181 = zext nneg i32 %29 to i64
  %182 = ptrtoint ptr addrspace(1) %1 to i64
  %183 = insertelement <2 x i64> poison, i64 %182, i64 0
  %184 = bitcast <2 x i64> %183 to <4 x i32>
  %185 = shufflevector <4 x i32> %184, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  %186 = ptrtoint ptr addrspace(1) %2 to i64
  %187 = insertelement <2 x i64> poison, i64 %186, i64 0
  %188 = bitcast <2 x i64> %187 to <4 x i32>
  %189 = shufflevector <4 x i32> %188, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  %190 = lshr i64 %181, 2
  %191 = and i64 %190, 48
  %192 = and i32 %29, 240
  %193 = zext nneg i32 %192 to i64
  %194 = sub nsw i64 %181, %193
  %195 = add nsw i64 %194, %191
  %196 = add nsw i64 %195, %24
  %197 = and i64 %181, 48
  %198 = add i64 %196, 1
  %199 = lshr i64 %198, 32
  %200 = trunc nuw i64 %199 to i32
  %201 = trunc i64 %198 to i32
  %202 = add i64 %196, 65
  %203 = lshr i64 %202, 32
  %204 = trunc nuw i64 %203 to i32
  %205 = trunc i64 %202 to i32
  %206 = insertelement <4 x i32> poison, i32 %205, i64 0
  %207 = shufflevector <4 x i32> %206, <4 x i32> poison, <4 x i32> zeroinitializer
  %208 = insertelement <4 x i32> poison, i32 %201, i64 0
  %209 = shufflevector <4 x i32> %208, <4 x i32> poison, <4 x i32> zeroinitializer
  br label %210

210:                                              ; preds = %180, %860
  %211 = phi [2 x <8 x float>] [ zeroinitializer, %180 ], [ %1796, %860 ]
  %212 = phi [2 x <8 x float>] [ zeroinitializer, %180 ], [ %1792, %860 ]
  %213 = phi [2 x <8 x float>] [ zeroinitializer, %180 ], [ %1788, %860 ]
  %214 = phi [2 x <1 x float>] [ [<1 x float> splat (float -inf), <1 x float> splat (float -inf)], %180 ], [ %1784, %860 ]
  %215 = phi [2 x <1 x float>] [ zeroinitializer, %180 ], [ %1410, %860 ]
  %216 = phi [2 x <8 x float>] [ zeroinitializer, %180 ], [ %1782, %860 ]
  %217 = phi i64 [ 0, %180 ], [ %1797, %860 ]
  %218 = shl nuw nsw i64 %217, 7
  %219 = lshr i64 %217, 12
  %220 = add nuw nsw i64 %219, %16
  %221 = lshr i64 %217, 5
  %222 = and i64 %221, 127
  %223 = shl i64 %220, 19
  %224 = and i64 %223, 16252928
  %225 = shl i64 %221, 19
  %226 = shl nuw nsw i64 %217, 14
  %227 = sub i64 %226, %225
  %228 = tail call i32 asm sideeffect "", "=v,0"(i32 %29) #9, !range !2, !frisk.register_lifetimes !3
  %229 = zext nneg i32 %228 to i64
  %230 = shl nuw nsw i64 %229, 3
  %231 = and i64 %230, 1536
  %232 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %231
  %233 = lshr i64 %229, 4
  %234 = and i64 %233, 8
  %235 = and i64 %229, 123
  %236 = lshr i64 %229, 2
  %237 = and i64 %236, 4
  %238 = add nuw nsw i64 %237, %229
  %239 = and i64 %238, 4
  %240 = or disjoint i64 %239, %235
  %241 = shl nuw nsw i64 %240, 7
  %242 = or disjoint i64 %241, %234
  %243 = or disjoint i64 %224, %222
  %244 = add i64 %243, %227
  %245 = add i64 %242, %244
  %246 = trunc i64 %245 to i32
  %247 = shl i32 %246, 1
  %248 = or disjoint i64 %229, 256
  %249 = shl nuw nsw i64 %248, 3
  %250 = and i64 %249, 3584
  %251 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %250
  %252 = lshr i64 %248, 4
  %253 = and i64 %252, 24
  %254 = or disjoint i64 %241, %253
  %255 = add i64 %254, %244
  %256 = trunc i64 %255 to i32
  %257 = shl i32 %256, 1
  %258 = add nuw nsw i64 %217, 1024
  %259 = lshr i64 %258, 12
  %260 = add nuw nsw i64 %259, %16
  %261 = add nsw i64 %221, %163
  %262 = shl i64 %260, 19
  %263 = and i64 %262, 16252928
  %264 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %231
  %265 = add i64 %227, %221
  %266 = add i64 %265, %163
  %267 = add i64 %266, %263
  %268 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %250
  %269 = insertelement <4 x ptr addrspace(3)> poison, ptr addrspace(3) %232, i64 0
  %270 = insertelement <4 x ptr addrspace(3)> %269, ptr addrspace(3) %251, i64 1
  %271 = insertelement <4 x ptr addrspace(3)> %270, ptr addrspace(3) %264, i64 2
  %272 = insertelement <4 x ptr addrspace(3)> %271, ptr addrspace(3) %268, i64 3
  %273 = ptrtoint <4 x ptr addrspace(3)> %272 to <4 x i64>
  %274 = trunc <4 x i64> %273 to <4 x i32>
  %275 = extractelement <4 x i32> %274, i64 0
  %276 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %275)
  %277 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %276)
  %278 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %277, i32 %247, <4 x i32> %185) #10
  %279 = extractelement <4 x i32> %274, i64 1
  %280 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %279)
  %281 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %280)
  %282 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %281, i32 %257, <4 x i32> %185) #10
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %283 = extractelement <4 x i32> %274, i64 2
  %284 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %283)
  %285 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %284)
  %286 = extractelement <4 x i32> %274, i64 3
  %287 = shl nuw nsw i64 %229, 1
  %288 = and i64 %287, 24
  %289 = shl i64 %229, 62
  %290 = ashr exact i64 %289, 57
  %291 = and i64 %290, -64
  %292 = and i64 %236, 12
  %293 = and i64 %229, 32
  %294 = icmp eq i64 %293, 0
  %295 = select i1 %294, i64 0, i64 1016
  %296 = mul nuw nsw i64 %229, 320
  %297 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %296
  %298 = mul nsw i64 %236, -1280
  %299 = getelementptr i8, ptr addrspace(3) %297, i64 %298
  %300 = getelementptr [2 x i8], ptr addrspace(3) %299, i64 %288
  %301 = getelementptr [2 x i8], ptr addrspace(3) %300, i64 %291
  %302 = getelementptr [2 x i8], ptr addrspace(3) %301, i64 %292
  %303 = getelementptr [2 x i8], ptr addrspace(3) %302, i64 %295
  %304 = getelementptr i8, ptr addrspace(3) %303, i64 4096
  %305 = shl nuw nsw i32 %228, 1
  %306 = and i32 %305, 24
  %307 = shl nuw nsw i32 %228, 8
  %308 = and i32 %307, 768
  %309 = or disjoint i32 %306, %308
  %310 = shl nuw nsw i32 %228, 5
  %311 = and i32 %310, 96
  %312 = or disjoint i32 %309, %311
  %313 = zext nneg i32 %312 to i64
  %314 = lshr i32 %228, 1
  %315 = and i32 %314, 1
  %316 = zext nneg i32 %315 to i64
  %317 = getelementptr i8, ptr addrspace(3) %303, i64 128
  %318 = getelementptr i8, ptr addrspace(3) %303, i64 4224
  %319 = getelementptr i8, ptr addrspace(3) %303, i64 1024
  %320 = getelementptr i8, ptr addrspace(3) %303, i64 5120
  %321 = getelementptr i8, ptr addrspace(3) %303, i64 1152
  %322 = getelementptr i8, ptr addrspace(3) %303, i64 5248
  %323 = add nuw nsw i64 %217, 2048
  %324 = lshr i64 %323, 12
  %325 = add nuw nsw i64 %324, %16
  %326 = insertelement <2 x i64> poison, i64 %261, i64 0
  %327 = shufflevector <2 x i64> %326, <2 x i64> poison, <2 x i32> zeroinitializer
  %328 = icmp slt <2 x i64> %327, <i64 -32, i64 -64>
  %329 = extractelement <2 x i1> %328, i64 0
  %330 = sext i1 %329 to i64
  %331 = add nsw <2 x i64> %327, <i64 32, i64 64>
  %332 = sub nsw <2 x i64> <i64 -33, i64 -65>, %327
  %333 = select <2 x i1> %328, <2 x i64> %332, <2 x i64> %331
  %334 = sdiv <2 x i64> %333, splat (i64 128)
  %335 = extractelement <2 x i64> %334, i64 0
  %336 = xor i64 %335, %330
  %337 = shl nsw i64 %336, 7
  %338 = sub i64 %267, %337
  %339 = add i64 %242, %338
  %340 = trunc i64 %339 to i32
  %341 = shl i32 %340, 1
  %342 = add i32 %341, 64
  %343 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %285, i32 %342, <4 x i32> %185) #10
  %344 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %286)
  %345 = add i64 %254, %338
  %346 = trunc i64 %345 to i32
  %347 = shl i32 %346, 1
  %348 = add i32 %347, 64
  %349 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %344)
  %350 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %349, i32 %348, <4 x i32> %185) #10
  %351 = load <4 x half>, ptr addrspace(3) %303, align 8
  %352 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %112, <4 x half> %351, <4 x float> zeroinitializer) #9
  %353 = load <4 x half>, ptr addrspace(3) %304, align 8
  %354 = tail call i64 asm sideeffect "", "=v,0"(i64 %171) #9, !frisk.register_lifetimes !3
  %355 = add nsw i64 %354, %316
  %356 = shl i64 %355, 6
  %357 = sub i64 %313, %356
  %358 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %357
  %359 = getelementptr [2 x i8], ptr addrspace(3) %358, i64 %292
  %360 = getelementptr [2 x i8], ptr addrspace(3) %359, i64 %295
  %361 = getelementptr i8, ptr addrspace(3) %360, i64 64
  %362 = load <4 x half>, ptr addrspace(3) %361, align 8
  %363 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %115, <4 x half> %353, <4 x float> %352) #9
  %364 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %112, <4 x half> %362, <4 x float> zeroinitializer) #9
  %365 = getelementptr i8, ptr addrspace(3) %360, i64 4160
  %366 = load <4 x half>, ptr addrspace(3) %365, align 8
  %367 = load <4 x half>, ptr addrspace(3) %317, align 8
  %368 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %115, <4 x half> %366, <4 x float> %364) #9
  %369 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %112, <4 x half> %367, <4 x float> zeroinitializer) #9
  %370 = load <4 x half>, ptr addrspace(3) %318, align 8
  %371 = getelementptr i8, ptr addrspace(3) %360, i64 192
  %372 = load <4 x half>, ptr addrspace(3) %371, align 8
  %373 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %115, <4 x half> %370, <4 x float> %369) #9
  %374 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %112, <4 x half> %372, <4 x float> zeroinitializer) #9
  %375 = getelementptr i8, ptr addrspace(3) %360, i64 4288
  %376 = load <4 x half>, ptr addrspace(3) %375, align 8
  %377 = load <4 x half>, ptr addrspace(3) %319, align 8
  %378 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %115, <4 x half> %376, <4 x float> %374) #9
  %379 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %112, <4 x half> %377, <4 x float> zeroinitializer) #9
  %380 = load <4 x half>, ptr addrspace(3) %320, align 8
  %381 = getelementptr i8, ptr addrspace(3) %360, i64 1088
  %382 = load <4 x half>, ptr addrspace(3) %381, align 8
  %383 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %115, <4 x half> %380, <4 x float> %379) #9
  %384 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %112, <4 x half> %382, <4 x float> zeroinitializer) #9
  %385 = getelementptr i8, ptr addrspace(3) %360, i64 5184
  %386 = load <4 x half>, ptr addrspace(3) %385, align 8
  %387 = load <4 x half>, ptr addrspace(3) %321, align 8
  %388 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %115, <4 x half> %386, <4 x float> %384) #9
  %389 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %112, <4 x half> %387, <4 x float> zeroinitializer) #9
  %390 = load <4 x half>, ptr addrspace(3) %322, align 8
  %391 = getelementptr i8, ptr addrspace(3) %360, i64 1216
  %392 = load <4 x half>, ptr addrspace(3) %391, align 8
  %393 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %115, <4 x half> %390, <4 x float> %389) #9
  %394 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %112, <4 x half> %392, <4 x float> zeroinitializer) #9
  %395 = getelementptr i8, ptr addrspace(3) %360, i64 5312
  %396 = load <4 x half>, ptr addrspace(3) %395, align 8
  %397 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %115, <4 x half> %396, <4 x float> %394) #9
  %398 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %118, <4 x half> %351, <4 x float> zeroinitializer) #9
  %399 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %119, <4 x half> %353, <4 x float> %398) #9
  %400 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %118, <4 x half> %362, <4 x float> zeroinitializer) #9
  %401 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %119, <4 x half> %366, <4 x float> %400) #9
  %402 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %118, <4 x half> %367, <4 x float> zeroinitializer) #9
  %403 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %119, <4 x half> %370, <4 x float> %402) #9
  %404 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %118, <4 x half> %372, <4 x float> zeroinitializer) #9
  %405 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %119, <4 x half> %376, <4 x float> %404) #9
  %406 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %118, <4 x half> %377, <4 x float> zeroinitializer) #9
  %407 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %119, <4 x half> %380, <4 x float> %406) #9
  %408 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %118, <4 x half> %382, <4 x float> zeroinitializer) #9
  %409 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %119, <4 x half> %386, <4 x float> %408) #9
  %410 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %118, <4 x half> %387, <4 x float> zeroinitializer) #9
  %411 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %119, <4 x half> %390, <4 x float> %410) #9
  %412 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %118, <4 x half> %392, <4 x float> zeroinitializer) #9
  %413 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %119, <4 x half> %396, <4 x float> %412) #9
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %414 = extractelement <2 x i1> %328, i64 1
  %415 = sext i1 %414 to i64
  %416 = extractelement <2 x i64> %334, i64 1
  %417 = xor i64 %416, %415
  %418 = shl i64 %325, 19
  %419 = and i64 %418, 16252928
  %420 = add i64 %266, %419
  %421 = shl nsw i64 %417, 7
  %422 = sub i64 %420, %421
  %423 = add i64 %242, %422
  %424 = trunc i64 %423 to i32
  %425 = shl i32 %424, 1
  %426 = add i32 %425, 128
  %427 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %277, i32 %426, <4 x i32> %185) #10
  %428 = add i64 %254, %422
  %429 = trunc i64 %428 to i32
  %430 = shl i32 %429, 1
  %431 = add i32 %430, 128
  %432 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %281, i32 %431, <4 x i32> %185) #10
  %433 = getelementptr i8, ptr addrspace(3) %303, i64 8192
  %434 = load <4 x half>, ptr addrspace(3) %433, align 8
  %435 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %130, <4 x half> %434, <4 x float> %363) #9
  %436 = getelementptr i8, ptr addrspace(3) %303, i64 12288
  %437 = load <4 x half>, ptr addrspace(3) %436, align 8
  %438 = getelementptr i8, ptr addrspace(3) %360, i64 8256
  %439 = load <4 x half>, ptr addrspace(3) %438, align 8
  %440 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %133, <4 x half> %437, <4 x float> %435) #9
  %441 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %130, <4 x half> %439, <4 x float> %368) #9
  %442 = getelementptr i8, ptr addrspace(3) %360, i64 12352
  %443 = load <4 x half>, ptr addrspace(3) %442, align 8
  %444 = getelementptr i8, ptr addrspace(3) %303, i64 8320
  %445 = load <4 x half>, ptr addrspace(3) %444, align 8
  %446 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %133, <4 x half> %443, <4 x float> %441) #9
  %447 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %130, <4 x half> %445, <4 x float> %373) #9
  %448 = getelementptr i8, ptr addrspace(3) %303, i64 12416
  %449 = load <4 x half>, ptr addrspace(3) %448, align 8
  %450 = getelementptr i8, ptr addrspace(3) %360, i64 8384
  %451 = load <4 x half>, ptr addrspace(3) %450, align 8
  %452 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %133, <4 x half> %449, <4 x float> %447) #9
  %453 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %130, <4 x half> %451, <4 x float> %378) #9
  %454 = getelementptr i8, ptr addrspace(3) %360, i64 12480
  %455 = load <4 x half>, ptr addrspace(3) %454, align 8
  %456 = getelementptr i8, ptr addrspace(3) %303, i64 9216
  %457 = load <4 x half>, ptr addrspace(3) %456, align 8
  %458 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %133, <4 x half> %455, <4 x float> %453) #9
  %459 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %130, <4 x half> %457, <4 x float> %383) #9
  %460 = getelementptr i8, ptr addrspace(3) %303, i64 13312
  %461 = load <4 x half>, ptr addrspace(3) %460, align 8
  %462 = getelementptr i8, ptr addrspace(3) %360, i64 9280
  %463 = load <4 x half>, ptr addrspace(3) %462, align 8
  %464 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %133, <4 x half> %461, <4 x float> %459) #9
  %465 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %130, <4 x half> %463, <4 x float> %388) #9
  %466 = getelementptr i8, ptr addrspace(3) %360, i64 13376
  %467 = load <4 x half>, ptr addrspace(3) %466, align 8
  %468 = getelementptr i8, ptr addrspace(3) %303, i64 9344
  %469 = load <4 x half>, ptr addrspace(3) %468, align 8
  %470 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %133, <4 x half> %467, <4 x float> %465) #9
  %471 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %130, <4 x half> %469, <4 x float> %393) #9
  %472 = getelementptr i8, ptr addrspace(3) %303, i64 13440
  %473 = load <4 x half>, ptr addrspace(3) %472, align 8
  %474 = getelementptr i8, ptr addrspace(3) %360, i64 9408
  %475 = load <4 x half>, ptr addrspace(3) %474, align 8
  %476 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %133, <4 x half> %473, <4 x float> %471) #9
  %477 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %130, <4 x half> %475, <4 x float> %397) #9
  %478 = getelementptr i8, ptr addrspace(3) %360, i64 13504
  %479 = load <4 x half>, ptr addrspace(3) %478, align 8
  %480 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %133, <4 x half> %479, <4 x float> %477) #9
  %481 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %134, <4 x half> %434, <4 x float> %399) #9
  %482 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %135, <4 x half> %437, <4 x float> %481) #9
  %483 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %134, <4 x half> %439, <4 x float> %401) #9
  %484 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %135, <4 x half> %443, <4 x float> %483) #9
  %485 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %134, <4 x half> %445, <4 x float> %403) #9
  %486 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %135, <4 x half> %449, <4 x float> %485) #9
  %487 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %134, <4 x half> %451, <4 x float> %405) #9
  %488 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %135, <4 x half> %455, <4 x float> %487) #9
  %489 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %134, <4 x half> %457, <4 x float> %407) #9
  %490 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %135, <4 x half> %461, <4 x float> %489) #9
  %491 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %134, <4 x half> %463, <4 x float> %409) #9
  %492 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %135, <4 x half> %467, <4 x float> %491) #9
  %493 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %134, <4 x half> %469, <4 x float> %411) #9
  %494 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %135, <4 x half> %473, <4 x float> %493) #9
  %495 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %134, <4 x half> %475, <4 x float> %413) #9
  %496 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %135, <4 x half> %479, <4 x float> %495) #9
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %497 = add nuw nsw i64 %217, 3072
  %498 = lshr i64 %497, 12
  %499 = add nuw nsw i64 %498, %16
  %500 = add nsw i64 %261, 96
  %501 = icmp slt i64 %261, -96
  %502 = sub nsw i64 -97, %261
  %503 = select i1 %501, i64 %502, i64 %500
  %504 = sdiv i64 %503, 128
  %505 = sext i1 %501 to i64
  %506 = xor i64 %504, %505
  %507 = shl i64 %499, 19
  %508 = and i64 %507, 16252928
  %509 = add i64 %266, %508
  %510 = shl nsw i64 %506, 7
  %511 = sub i64 %509, %510
  %512 = add i64 %242, %511
  %513 = trunc i64 %512 to i32
  %514 = shl i32 %513, 1
  %515 = add i32 %514, 192
  %516 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %285, i32 %515, <4 x i32> %185) #10
  %517 = add i64 %254, %511
  %518 = trunc i64 %517 to i32
  %519 = shl i32 %518, 1
  %520 = add i32 %519, 192
  %521 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %349, i32 %520, <4 x i32> %185) #10
  %522 = load <4 x half>, ptr addrspace(3) %303, align 8
  %523 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %143, <4 x half> %522, <4 x float> %440) #9
  %524 = load <4 x half>, ptr addrspace(3) %304, align 8
  %525 = load <4 x half>, ptr addrspace(3) %361, align 8
  %526 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %145, <4 x half> %524, <4 x float> %523) #9
  %527 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %143, <4 x half> %525, <4 x float> %446) #9
  %528 = load <4 x half>, ptr addrspace(3) %365, align 8
  %529 = load <4 x half>, ptr addrspace(3) %317, align 8
  %530 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %145, <4 x half> %528, <4 x float> %527) #9
  %531 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %143, <4 x half> %529, <4 x float> %452) #9
  %532 = load <4 x half>, ptr addrspace(3) %318, align 8
  %533 = load <4 x half>, ptr addrspace(3) %371, align 8
  %534 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %145, <4 x half> %532, <4 x float> %531) #9
  %535 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %143, <4 x half> %533, <4 x float> %458) #9
  %536 = load <4 x half>, ptr addrspace(3) %375, align 8
  %537 = load <4 x half>, ptr addrspace(3) %319, align 8
  %538 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %145, <4 x half> %536, <4 x float> %535) #9
  %539 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %143, <4 x half> %537, <4 x float> %464) #9
  %540 = load <4 x half>, ptr addrspace(3) %320, align 8
  %541 = load <4 x half>, ptr addrspace(3) %381, align 8
  %542 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %145, <4 x half> %540, <4 x float> %539) #9
  %543 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %143, <4 x half> %541, <4 x float> %470) #9
  %544 = load <4 x half>, ptr addrspace(3) %385, align 8
  %545 = load <4 x half>, ptr addrspace(3) %321, align 8
  %546 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %145, <4 x half> %544, <4 x float> %543) #9
  %547 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %143, <4 x half> %545, <4 x float> %476) #9
  %548 = load <4 x half>, ptr addrspace(3) %322, align 8
  %549 = load <4 x half>, ptr addrspace(3) %391, align 8
  %550 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %145, <4 x half> %548, <4 x float> %547) #9
  %551 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %143, <4 x half> %549, <4 x float> %480) #9
  %552 = load <4 x half>, ptr addrspace(3) %395, align 8
  %553 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %145, <4 x half> %552, <4 x float> %551) #9
  %554 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %146, <4 x half> %522, <4 x float> %482) #9
  %555 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %147, <4 x half> %524, <4 x float> %554) #9
  %556 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %146, <4 x half> %525, <4 x float> %484) #9
  %557 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %147, <4 x half> %528, <4 x float> %556) #9
  %558 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %146, <4 x half> %529, <4 x float> %486) #9
  %559 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %147, <4 x half> %532, <4 x float> %558) #9
  %560 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %146, <4 x half> %533, <4 x float> %488) #9
  %561 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %147, <4 x half> %536, <4 x float> %560) #9
  %562 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %146, <4 x half> %537, <4 x float> %490) #9
  %563 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %147, <4 x half> %540, <4 x float> %562) #9
  %564 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %146, <4 x half> %541, <4 x float> %492) #9
  %565 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %147, <4 x half> %544, <4 x float> %564) #9
  %566 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %146, <4 x half> %545, <4 x float> %494) #9
  %567 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %147, <4 x half> %548, <4 x float> %566) #9
  %568 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %146, <4 x half> %549, <4 x float> %496) #9
  %569 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %147, <4 x half> %552, <4 x float> %568) #9
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %570 = add nuw nsw i64 %221, %16
  %571 = shl i64 %570, 19
  %572 = and i64 %571, 16252928
  %573 = shl nuw nsw i64 %229, 6
  %574 = and i64 %573, 16128
  %575 = and i64 %230, 24
  %576 = add i64 %572, %227
  %577 = or disjoint i64 %575, %576
  %578 = shl i64 %577, 1
  %579 = or disjoint i64 %578, %574
  %580 = trunc i64 %579 to i32
  %581 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %277, i32 %580, <4 x i32> %189) #10
  %582 = shl nuw nsw i64 %248, 6
  %583 = and i64 %582, 32512
  %584 = or disjoint i64 %578, %583
  %585 = trunc i64 %584 to i32
  %586 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %281, i32 %585, <4 x i32> %189) #10
  %587 = load <4 x half>, ptr addrspace(3) %433, align 8
  %588 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %151, <4 x half> %587, <4 x float> %526) #9
  %589 = load <4 x half>, ptr addrspace(3) %436, align 8
  %590 = load <4 x half>, ptr addrspace(3) %438, align 8
  %591 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %153, <4 x half> %589, <4 x float> %588) #9
  %592 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %151, <4 x half> %590, <4 x float> %530) #9
  %593 = load <4 x half>, ptr addrspace(3) %442, align 8
  %594 = load <4 x half>, ptr addrspace(3) %444, align 8
  %595 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %153, <4 x half> %593, <4 x float> %592) #9
  %596 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %151, <4 x half> %594, <4 x float> %534) #9
  %597 = load <4 x half>, ptr addrspace(3) %448, align 8
  %598 = load <4 x half>, ptr addrspace(3) %450, align 8
  %599 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %153, <4 x half> %597, <4 x float> %596) #9
  %600 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %151, <4 x half> %598, <4 x float> %538) #9
  %601 = load <4 x half>, ptr addrspace(3) %454, align 8
  %602 = load <4 x half>, ptr addrspace(3) %456, align 8
  %603 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %153, <4 x half> %601, <4 x float> %600) #9
  %604 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %151, <4 x half> %602, <4 x float> %542) #9
  %605 = load <4 x half>, ptr addrspace(3) %460, align 8
  %606 = load <4 x half>, ptr addrspace(3) %462, align 8
  %607 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %153, <4 x half> %605, <4 x float> %604) #9
  %608 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %151, <4 x half> %606, <4 x float> %546) #9
  %609 = load <4 x half>, ptr addrspace(3) %466, align 8
  %610 = load <4 x half>, ptr addrspace(3) %468, align 8
  %611 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %153, <4 x half> %609, <4 x float> %608) #9
  %612 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %151, <4 x half> %610, <4 x float> %550) #9
  %613 = load <4 x half>, ptr addrspace(3) %472, align 8
  %614 = load <4 x half>, ptr addrspace(3) %474, align 8
  %615 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %153, <4 x half> %613, <4 x float> %612) #9
  %616 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %151, <4 x half> %614, <4 x float> %553) #9
  %617 = load <4 x half>, ptr addrspace(3) %478, align 8
  %618 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %153, <4 x half> %617, <4 x float> %616) #9
  %619 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %154, <4 x half> %587, <4 x float> %555) #9
  %620 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %155, <4 x half> %589, <4 x float> %619) #9
  %621 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %154, <4 x half> %590, <4 x float> %557) #9
  %622 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %155, <4 x half> %593, <4 x float> %621) #9
  %623 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %154, <4 x half> %594, <4 x float> %559) #9
  %624 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %155, <4 x half> %597, <4 x float> %623) #9
  %625 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %154, <4 x half> %598, <4 x float> %561) #9
  %626 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %155, <4 x half> %601, <4 x float> %625) #9
  %627 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %154, <4 x half> %602, <4 x float> %563) #9
  %628 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %155, <4 x half> %605, <4 x float> %627) #9
  %629 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %154, <4 x half> %606, <4 x float> %565) #9
  %630 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %155, <4 x half> %609, <4 x float> %629) #9
  %631 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %154, <4 x half> %610, <4 x float> %567) #9
  %632 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %155, <4 x half> %613, <4 x float> %631) #9
  %633 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %154, <4 x half> %614, <4 x float> %569) #9
  %634 = or disjoint i64 %218, 127
  %635 = icmp uge i64 %24, %634
  %636 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %155, <4 x half> %617, <4 x float> %633) #9
  %637 = and i1 %158, %635
  br i1 %637, label %860, label %638

638:                                              ; preds = %210
  %639 = or disjoint i64 %218, %197
  %640 = lshr i64 %217, 25
  %641 = trunc nuw i64 %640 to i32
  %642 = trunc i64 %639 to i32
  %643 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %642, i32 1) #9, !frisk.register_lifetimes !3
  %644 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %642, i32 2) #9, !frisk.register_lifetimes !3
  %645 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %642, i32 3) #9, !frisk.register_lifetimes !3
  %646 = insertelement <4 x i32> poison, i32 %642, i64 0
  %647 = insertelement <4 x i32> %646, i32 %643, i64 1
  %648 = insertelement <4 x i32> %647, i32 %644, i64 2
  %649 = insertelement <4 x i32> %648, i32 %645, i64 3
  %650 = or <4 x i32> %649, <i32 0, i32 1, i32 2, i32 3>
  %651 = icmp ugt <4 x i32> %209, %650
  %652 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %642, i32 4) #9, !frisk.register_lifetimes !3
  %653 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %642, i32 5) #9, !frisk.register_lifetimes !3
  %654 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %642, i32 6) #9, !frisk.register_lifetimes !3
  %655 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %642, i32 7) #9, !frisk.register_lifetimes !3
  %656 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %642, i32 8) #9, !frisk.register_lifetimes !3
  %657 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %642, i32 9) #9, !frisk.register_lifetimes !3
  %658 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %642, i32 10) #9, !frisk.register_lifetimes !3
  %659 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %642, i32 11) #9, !frisk.register_lifetimes !3
  %660 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %642, i32 12) #9, !frisk.register_lifetimes !3
  %661 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %642, i32 13) #9, !frisk.register_lifetimes !3
  %662 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %642, i32 14) #9, !frisk.register_lifetimes !3
  %663 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %642, i32 15) #9, !frisk.register_lifetimes !3
  %664 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %642, i32 16) #9, !frisk.register_lifetimes !3
  %665 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %642, i32 17) #9, !frisk.register_lifetimes !3
  %666 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %642, i32 18) #9, !frisk.register_lifetimes !3
  %667 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %642, i32 19) #9, !frisk.register_lifetimes !3
  %668 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %642, i32 20) #9, !frisk.register_lifetimes !3
  %669 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %642, i32 21) #9, !frisk.register_lifetimes !3
  %670 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %642, i32 22) #9, !frisk.register_lifetimes !3
  %671 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %642, i32 23) #9, !frisk.register_lifetimes !3
  %672 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %642, i32 24) #9, !frisk.register_lifetimes !3
  %673 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %642, i32 25) #9, !frisk.register_lifetimes !3
  %674 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %642, i32 26) #9, !frisk.register_lifetimes !3
  %675 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %642, i32 27) #9, !frisk.register_lifetimes !3
  %676 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %642, i32 28) #9, !frisk.register_lifetimes !3
  %677 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %642, i32 29) #9, !frisk.register_lifetimes !3
  %678 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %642, i32 30) #9, !frisk.register_lifetimes !3
  %679 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %642, i32 31) #9, !frisk.register_lifetimes !3
  %680 = insertelement <4 x i32> poison, i32 %672, i64 0
  %681 = insertelement <4 x i32> %680, i32 %673, i64 1
  %682 = insertelement <4 x i32> %681, i32 %674, i64 2
  %683 = insertelement <4 x i32> %682, i32 %675, i64 3
  %684 = or <4 x i32> %683, <i32 72, i32 73, i32 74, i32 75>
  %685 = icmp ult <4 x i32> %684, %209
  %686 = insertelement <4 x i32> poison, i32 %668, i64 0
  %687 = insertelement <4 x i32> %686, i32 %669, i64 1
  %688 = insertelement <4 x i32> %687, i32 %670, i64 2
  %689 = insertelement <4 x i32> %688, i32 %671, i64 3
  %690 = or <4 x i32> %689, <i32 68, i32 69, i32 70, i32 71>
  %691 = icmp ult <4 x i32> %690, %209
  %692 = insertelement <4 x i32> poison, i32 %664, i64 0
  %693 = insertelement <4 x i32> %692, i32 %665, i64 1
  %694 = insertelement <4 x i32> %693, i32 %666, i64 2
  %695 = insertelement <4 x i32> %694, i32 %667, i64 3
  %696 = or <4 x i32> %695, <i32 64, i32 65, i32 66, i32 67>
  %697 = icmp ult <4 x i32> %696, %209
  %698 = insertelement <4 x i32> poison, i32 %660, i64 0
  %699 = insertelement <4 x i32> %698, i32 %661, i64 1
  %700 = insertelement <4 x i32> %699, i32 %662, i64 2
  %701 = insertelement <4 x i32> %700, i32 %663, i64 3
  %702 = or <4 x i32> %701, <i32 12, i32 13, i32 14, i32 15>
  %703 = icmp ult <4 x i32> %702, %209
  %704 = insertelement <4 x i32> poison, i32 %656, i64 0
  %705 = insertelement <4 x i32> %704, i32 %657, i64 1
  %706 = insertelement <4 x i32> %705, i32 %658, i64 2
  %707 = insertelement <4 x i32> %706, i32 %659, i64 3
  %708 = or <4 x i32> %707, <i32 8, i32 9, i32 10, i32 11>
  %709 = icmp ult <4 x i32> %708, %209
  %710 = insertelement <4 x i32> poison, i32 %652, i64 0
  %711 = insertelement <4 x i32> %710, i32 %653, i64 1
  %712 = insertelement <4 x i32> %711, i32 %654, i64 2
  %713 = insertelement <4 x i32> %712, i32 %655, i64 3
  %714 = or <4 x i32> %713, <i32 4, i32 5, i32 6, i32 7>
  %715 = icmp ult <4 x i32> %714, %209
  %716 = icmp eq i32 %200, %641
  %717 = icmp ugt i32 %200, %641
  %718 = insertelement <4 x i1> poison, i1 %716, i64 0
  %719 = shufflevector <4 x i1> %718, <4 x i1> poison, <4 x i32> zeroinitializer
  %720 = insertelement <4 x i1> poison, i1 %717, i64 0
  %721 = shufflevector <4 x i1> %720, <4 x i1> poison, <4 x i32> zeroinitializer
  %722 = select <4 x i1> %719, <4 x i1> %651, <4 x i1> %721
  %723 = select <4 x i1> %722, <4 x float> zeroinitializer, <4 x float> splat (float -inf)
  %724 = select <4 x i1> %719, <4 x i1> %715, <4 x i1> %721
  %725 = select <4 x i1> %724, <4 x float> zeroinitializer, <4 x float> splat (float -inf)
  %726 = select <4 x i1> %719, <4 x i1> %709, <4 x i1> %721
  %727 = select <4 x i1> %726, <4 x float> zeroinitializer, <4 x float> splat (float -inf)
  %728 = select <4 x i1> %719, <4 x i1> %703, <4 x i1> %721
  %729 = select <4 x i1> %728, <4 x float> zeroinitializer, <4 x float> splat (float -inf)
  %730 = select <4 x i1> %719, <4 x i1> %697, <4 x i1> %721
  %731 = select <4 x i1> %730, <4 x float> zeroinitializer, <4 x float> splat (float -inf)
  %732 = select <4 x i1> %719, <4 x i1> %691, <4 x i1> %721
  %733 = select <4 x i1> %732, <4 x float> zeroinitializer, <4 x float> splat (float -inf)
  %734 = select <4 x i1> %719, <4 x i1> %685, <4 x i1> %721
  %735 = select <4 x i1> %734, <4 x float> zeroinitializer, <4 x float> splat (float -inf)
  %736 = insertelement <4 x i32> poison, i32 %676, i64 0
  %737 = insertelement <4 x i32> %736, i32 %677, i64 1
  %738 = insertelement <4 x i32> %737, i32 %678, i64 2
  %739 = insertelement <4 x i32> %738, i32 %679, i64 3
  %740 = or <4 x i32> %739, <i32 76, i32 77, i32 78, i32 79>
  %741 = icmp ult <4 x i32> %740, %209
  %742 = select <4 x i1> %719, <4 x i1> %741, <4 x i1> %721
  %743 = select <4 x i1> %742, <4 x float> zeroinitializer, <4 x float> splat (float -inf)
  %744 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %642, i32 32) #9, !frisk.register_lifetimes !3
  %745 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %642, i32 33) #9, !frisk.register_lifetimes !3
  %746 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %642, i32 34) #9, !frisk.register_lifetimes !3
  %747 = insertelement <4 x i32> %646, i32 %744, i64 1
  %748 = insertelement <4 x i32> %747, i32 %745, i64 2
  %749 = insertelement <4 x i32> %748, i32 %746, i64 3
  %750 = or <4 x i32> %749, <i32 0, i32 1, i32 2, i32 3>
  %751 = icmp ugt <4 x i32> %207, %750
  %752 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %642, i32 35) #9, !frisk.register_lifetimes !3
  %753 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %642, i32 36) #9, !frisk.register_lifetimes !3
  %754 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %642, i32 37) #9, !frisk.register_lifetimes !3
  %755 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %642, i32 38) #9, !frisk.register_lifetimes !3
  %756 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %642, i32 39) #9, !frisk.register_lifetimes !3
  %757 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %642, i32 40) #9, !frisk.register_lifetimes !3
  %758 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %642, i32 41) #9, !frisk.register_lifetimes !3
  %759 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %642, i32 42) #9, !frisk.register_lifetimes !3
  %760 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %642, i32 43) #9, !frisk.register_lifetimes !3
  %761 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %642, i32 44) #9, !frisk.register_lifetimes !3
  %762 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %642, i32 45) #9, !frisk.register_lifetimes !3
  %763 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %642, i32 46) #9, !frisk.register_lifetimes !3
  %764 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %642, i32 47) #9, !frisk.register_lifetimes !3
  %765 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %642, i32 48) #9, !frisk.register_lifetimes !3
  %766 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %642, i32 49) #9, !frisk.register_lifetimes !3
  %767 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %642, i32 50) #9, !frisk.register_lifetimes !3
  %768 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %642, i32 51) #9, !frisk.register_lifetimes !3
  %769 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %642, i32 52) #9, !frisk.register_lifetimes !3
  %770 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %642, i32 53) #9, !frisk.register_lifetimes !3
  %771 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %642, i32 54) #9, !frisk.register_lifetimes !3
  %772 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %642, i32 55) #9, !frisk.register_lifetimes !3
  %773 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %642, i32 56) #9, !frisk.register_lifetimes !3
  %774 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %642, i32 57) #9, !frisk.register_lifetimes !3
  %775 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %642, i32 58) #9, !frisk.register_lifetimes !3
  %776 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %642, i32 59) #9, !frisk.register_lifetimes !3
  %777 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %642, i32 60) #9, !frisk.register_lifetimes !3
  %778 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %642, i32 61) #9, !frisk.register_lifetimes !3
  %779 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %642, i32 62) #9, !frisk.register_lifetimes !3
  %780 = insertelement <4 x i32> poison, i32 %772, i64 0
  %781 = insertelement <4 x i32> %780, i32 %773, i64 1
  %782 = insertelement <4 x i32> %781, i32 %774, i64 2
  %783 = insertelement <4 x i32> %782, i32 %775, i64 3
  %784 = or <4 x i32> %783, <i32 72, i32 73, i32 74, i32 75>
  %785 = icmp ult <4 x i32> %784, %207
  %786 = insertelement <4 x i32> poison, i32 %768, i64 0
  %787 = insertelement <4 x i32> %786, i32 %769, i64 1
  %788 = insertelement <4 x i32> %787, i32 %770, i64 2
  %789 = insertelement <4 x i32> %788, i32 %771, i64 3
  %790 = or <4 x i32> %789, <i32 68, i32 69, i32 70, i32 71>
  %791 = icmp ult <4 x i32> %790, %207
  %792 = insertelement <4 x i32> poison, i32 %764, i64 0
  %793 = insertelement <4 x i32> %792, i32 %765, i64 1
  %794 = insertelement <4 x i32> %793, i32 %766, i64 2
  %795 = insertelement <4 x i32> %794, i32 %767, i64 3
  %796 = or <4 x i32> %795, <i32 64, i32 65, i32 66, i32 67>
  %797 = icmp ult <4 x i32> %796, %207
  %798 = insertelement <4 x i32> poison, i32 %760, i64 0
  %799 = insertelement <4 x i32> %798, i32 %761, i64 1
  %800 = insertelement <4 x i32> %799, i32 %762, i64 2
  %801 = insertelement <4 x i32> %800, i32 %763, i64 3
  %802 = or <4 x i32> %801, <i32 12, i32 13, i32 14, i32 15>
  %803 = icmp ult <4 x i32> %802, %207
  %804 = insertelement <4 x i32> poison, i32 %756, i64 0
  %805 = insertelement <4 x i32> %804, i32 %757, i64 1
  %806 = insertelement <4 x i32> %805, i32 %758, i64 2
  %807 = insertelement <4 x i32> %806, i32 %759, i64 3
  %808 = or <4 x i32> %807, <i32 8, i32 9, i32 10, i32 11>
  %809 = icmp ult <4 x i32> %808, %207
  %810 = insertelement <4 x i32> poison, i32 %752, i64 0
  %811 = insertelement <4 x i32> %810, i32 %753, i64 1
  %812 = insertelement <4 x i32> %811, i32 %754, i64 2
  %813 = insertelement <4 x i32> %812, i32 %755, i64 3
  %814 = or <4 x i32> %813, <i32 4, i32 5, i32 6, i32 7>
  %815 = icmp ult <4 x i32> %814, %207
  %816 = icmp eq i32 %204, %641
  %817 = icmp ugt i32 %204, %641
  %818 = insertelement <4 x i1> poison, i1 %816, i64 0
  %819 = shufflevector <4 x i1> %818, <4 x i1> poison, <4 x i32> zeroinitializer
  %820 = insertelement <4 x i1> poison, i1 %817, i64 0
  %821 = shufflevector <4 x i1> %820, <4 x i1> poison, <4 x i32> zeroinitializer
  %822 = select <4 x i1> %819, <4 x i1> %751, <4 x i1> %821
  %823 = select <4 x i1> %822, <4 x float> zeroinitializer, <4 x float> splat (float -inf)
  %824 = select <4 x i1> %819, <4 x i1> %815, <4 x i1> %821
  %825 = select <4 x i1> %824, <4 x float> zeroinitializer, <4 x float> splat (float -inf)
  %826 = select <4 x i1> %819, <4 x i1> %809, <4 x i1> %821
  %827 = select <4 x i1> %826, <4 x float> zeroinitializer, <4 x float> splat (float -inf)
  %828 = select <4 x i1> %819, <4 x i1> %803, <4 x i1> %821
  %829 = select <4 x i1> %828, <4 x float> zeroinitializer, <4 x float> splat (float -inf)
  %830 = select <4 x i1> %819, <4 x i1> %797, <4 x i1> %821
  %831 = select <4 x i1> %830, <4 x float> zeroinitializer, <4 x float> splat (float -inf)
  %832 = select <4 x i1> %819, <4 x i1> %791, <4 x i1> %821
  %833 = select <4 x i1> %832, <4 x float> zeroinitializer, <4 x float> splat (float -inf)
  %834 = select <4 x i1> %819, <4 x i1> %785, <4 x i1> %821
  %835 = select <4 x i1> %834, <4 x float> zeroinitializer, <4 x float> splat (float -inf)
  %836 = insertelement <4 x i32> poison, i32 %776, i64 0
  %837 = insertelement <4 x i32> %836, i32 %777, i64 1
  %838 = insertelement <4 x i32> %837, i32 %778, i64 2
  %839 = insertelement <4 x i32> %838, i32 %779, i64 3
  %840 = or <4 x i32> %839, <i32 76, i32 77, i32 78, i32 79>
  %841 = icmp ult <4 x i32> %840, %207
  %842 = select <4 x i1> %819, <4 x i1> %841, <4 x i1> %821
  %843 = select <4 x i1> %842, <4 x float> zeroinitializer, <4 x float> splat (float -inf)
  %844 = fadd <4 x float> %591, %723
  %845 = fadd <4 x float> %595, %725
  %846 = fadd <4 x float> %599, %727
  %847 = fadd <4 x float> %603, %729
  %848 = fadd <4 x float> %607, %731
  %849 = fadd <4 x float> %611, %733
  %850 = fadd <4 x float> %615, %735
  %851 = fadd <4 x float> %618, %743
  %852 = fadd <4 x float> %620, %823
  %853 = fadd <4 x float> %622, %825
  %854 = fadd <4 x float> %624, %827
  %855 = fadd <4 x float> %626, %829
  %856 = fadd <4 x float> %628, %831
  %857 = fadd <4 x float> %630, %833
  %858 = fadd <4 x float> %632, %835
  %859 = fadd <4 x float> %636, %843
  br label %860

860:                                              ; preds = %638, %210
  %861 = phi <4 x float> [ %844, %638 ], [ %591, %210 ]
  %862 = phi <4 x float> [ %845, %638 ], [ %595, %210 ]
  %863 = phi <4 x float> [ %846, %638 ], [ %599, %210 ]
  %864 = phi <4 x float> [ %847, %638 ], [ %603, %210 ]
  %865 = phi <4 x float> [ %848, %638 ], [ %607, %210 ]
  %866 = phi <4 x float> [ %849, %638 ], [ %611, %210 ]
  %867 = phi <4 x float> [ %850, %638 ], [ %615, %210 ]
  %868 = phi <4 x float> [ %851, %638 ], [ %618, %210 ]
  %869 = phi <4 x float> [ %852, %638 ], [ %620, %210 ]
  %870 = phi <4 x float> [ %853, %638 ], [ %622, %210 ]
  %871 = phi <4 x float> [ %854, %638 ], [ %624, %210 ]
  %872 = phi <4 x float> [ %855, %638 ], [ %626, %210 ]
  %873 = phi <4 x float> [ %856, %638 ], [ %628, %210 ]
  %874 = phi <4 x float> [ %857, %638 ], [ %630, %210 ]
  %875 = phi <4 x float> [ %858, %638 ], [ %632, %210 ]
  %876 = phi <4 x float> [ %859, %638 ], [ %636, %210 ]
  %877 = extractelement <4 x float> %861, i64 0
  %878 = extractelement <4 x float> %861, i64 1
  %879 = extractelement <4 x float> %861, i64 2
  %880 = extractelement <4 x float> %861, i64 3
  %881 = extractelement <4 x float> %862, i64 0
  %882 = extractelement <4 x float> %862, i64 1
  %883 = extractelement <4 x float> %862, i64 2
  %884 = extractelement <4 x float> %862, i64 3
  %885 = extractelement <4 x float> %863, i64 0
  %886 = extractelement <4 x float> %863, i64 1
  %887 = extractelement <4 x float> %863, i64 2
  %888 = extractelement <4 x float> %863, i64 3
  %889 = extractelement <4 x float> %864, i64 0
  %890 = extractelement <4 x float> %864, i64 1
  %891 = extractelement <4 x float> %864, i64 2
  %892 = extractelement <4 x float> %864, i64 3
  %893 = extractelement <4 x float> %865, i64 0
  %894 = extractelement <4 x float> %865, i64 1
  %895 = extractelement <4 x float> %865, i64 2
  %896 = extractelement <4 x float> %865, i64 3
  %897 = extractelement <4 x float> %866, i64 0
  %898 = extractelement <4 x float> %866, i64 1
  %899 = extractelement <4 x float> %866, i64 2
  %900 = extractelement <4 x float> %866, i64 3
  %901 = extractelement <4 x float> %867, i64 0
  %902 = extractelement <4 x float> %867, i64 1
  %903 = extractelement <4 x float> %867, i64 2
  %904 = extractelement <4 x float> %867, i64 3
  %905 = extractelement <4 x float> %868, i64 0
  %906 = extractelement <4 x float> %868, i64 1
  %907 = extractelement <4 x float> %868, i64 2
  %908 = extractelement <4 x float> %868, i64 3
  %909 = extractelement <4 x float> %869, i64 0
  %910 = extractelement <4 x float> %869, i64 1
  %911 = extractelement <4 x float> %869, i64 2
  %912 = extractelement <4 x float> %869, i64 3
  %913 = extractelement <4 x float> %870, i64 0
  %914 = extractelement <4 x float> %870, i64 1
  %915 = extractelement <4 x float> %870, i64 2
  %916 = extractelement <4 x float> %870, i64 3
  %917 = extractelement <4 x float> %871, i64 0
  %918 = extractelement <4 x float> %871, i64 1
  %919 = extractelement <4 x float> %871, i64 2
  %920 = extractelement <4 x float> %871, i64 3
  %921 = extractelement <4 x float> %872, i64 0
  %922 = extractelement <4 x float> %872, i64 1
  %923 = extractelement <4 x float> %872, i64 2
  %924 = extractelement <4 x float> %872, i64 3
  %925 = extractelement <4 x float> %873, i64 0
  %926 = extractelement <4 x float> %873, i64 1
  %927 = extractelement <4 x float> %873, i64 2
  %928 = extractelement <4 x float> %873, i64 3
  %929 = extractelement <4 x float> %874, i64 0
  %930 = extractelement <4 x float> %874, i64 1
  %931 = extractelement <4 x float> %874, i64 2
  %932 = extractelement <4 x float> %874, i64 3
  %933 = extractelement <4 x float> %875, i64 0
  %934 = extractelement <4 x float> %875, i64 1
  %935 = extractelement <4 x float> %875, i64 2
  %936 = extractelement <4 x float> %875, i64 3
  %937 = extractelement <4 x float> %876, i64 0
  %938 = extractelement <4 x float> %876, i64 1
  %939 = extractelement <4 x float> %876, i64 2
  %940 = extractelement <4 x float> %876, i64 3
  %941 = tail call float @llvm.maxnum.f32(float %877, float -inf)
  %942 = tail call float @llvm.maxnum.f32(float %941, float %881)
  %943 = tail call float @llvm.maxnum.f32(float %878, float %882)
  %944 = tail call float @llvm.maxnum.f32(float %879, float %883)
  %945 = tail call float @llvm.maxnum.f32(float %880, float %884)
  %946 = tail call float @llvm.maxnum.f32(float %942, float %885)
  %947 = tail call float @llvm.maxnum.f32(float %943, float %886)
  %948 = tail call float @llvm.maxnum.f32(float %944, float %887)
  %949 = tail call float @llvm.maxnum.f32(float %945, float %888)
  %950 = tail call float @llvm.maxnum.f32(float %946, float %889)
  %951 = tail call float @llvm.maxnum.f32(float %947, float %890)
  %952 = tail call float @llvm.maxnum.f32(float %948, float %891)
  %953 = tail call float @llvm.maxnum.f32(float %949, float %892)
  %954 = tail call float @llvm.maxnum.f32(float %950, float %893)
  %955 = tail call float @llvm.maxnum.f32(float %951, float %894)
  %956 = tail call float @llvm.maxnum.f32(float %952, float %895)
  %957 = tail call float @llvm.maxnum.f32(float %953, float %896)
  %958 = tail call float @llvm.maxnum.f32(float %954, float %897)
  %959 = tail call float @llvm.maxnum.f32(float %955, float %898)
  %960 = tail call float @llvm.maxnum.f32(float %956, float %899)
  %961 = tail call float @llvm.maxnum.f32(float %957, float %900)
  %962 = tail call float @llvm.maxnum.f32(float %958, float %901)
  %963 = tail call float @llvm.maxnum.f32(float %959, float %902)
  %964 = tail call float @llvm.maxnum.f32(float %960, float %903)
  %965 = tail call float @llvm.maxnum.f32(float %961, float %904)
  %966 = tail call float @llvm.maxnum.f32(float %962, float %905)
  %967 = tail call float @llvm.maxnum.f32(float %963, float %906)
  %968 = tail call float @llvm.maxnum.f32(float %964, float %907)
  %969 = tail call float @llvm.maxnum.f32(float %965, float %908)
  %970 = tail call float @llvm.maxnum.f32(float %966, float %967)
  %971 = tail call float @llvm.maxnum.f32(float %968, float %969)
  %972 = tail call float @llvm.maxnum.f32(float %970, float %971)
  %973 = tail call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %974 = tail call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %973)
  %975 = add nuw nsw i32 %974, 64
  %976 = xor i32 %974, 16
  %977 = and i32 %975, 192
  %978 = icmp samesign ult i32 %976, %977
  %979 = select i1 %978, i32 %976, i32 %974
  %980 = shl nuw nsw i32 %979, 2
  %981 = bitcast float %972 to i32
  %982 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %980, i32 %981)
  %983 = bitcast i32 %982 to float
  %984 = tail call float @llvm.maxnum.f32(float %972, float %983)
  %985 = xor i32 %974, 32
  %986 = icmp samesign ult i32 %985, %977
  %987 = select i1 %986, i32 %985, i32 %974
  %988 = shl nuw nsw i32 %987, 2
  %989 = bitcast float %984 to i32
  %990 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %988, i32 %989)
  %991 = bitcast i32 %990 to float
  %992 = tail call float @llvm.maxnum.f32(float %984, float %991)
  %993 = extractvalue [2 x <1 x float>] %214, 0
  %994 = insertelement <1 x float> poison, float %992, i64 0
  %995 = tail call <1 x float> @llvm.maxnum.v1f32(<1 x float> %994, <1 x float> %993)
  %996 = tail call float @llvm.maxnum.f32(float %909, float -inf)
  %997 = tail call float @llvm.maxnum.f32(float %996, float %913)
  %998 = tail call float @llvm.maxnum.f32(float %910, float %914)
  %999 = tail call float @llvm.maxnum.f32(float %911, float %915)
  %1000 = tail call float @llvm.maxnum.f32(float %912, float %916)
  %1001 = tail call float @llvm.maxnum.f32(float %997, float %917)
  %1002 = tail call float @llvm.maxnum.f32(float %998, float %918)
  %1003 = tail call float @llvm.maxnum.f32(float %999, float %919)
  %1004 = tail call float @llvm.maxnum.f32(float %1000, float %920)
  %1005 = tail call float @llvm.maxnum.f32(float %1001, float %921)
  %1006 = tail call float @llvm.maxnum.f32(float %1002, float %922)
  %1007 = tail call float @llvm.maxnum.f32(float %1003, float %923)
  %1008 = tail call float @llvm.maxnum.f32(float %1004, float %924)
  %1009 = tail call float @llvm.maxnum.f32(float %1005, float %925)
  %1010 = tail call float @llvm.maxnum.f32(float %1006, float %926)
  %1011 = tail call float @llvm.maxnum.f32(float %1007, float %927)
  %1012 = tail call float @llvm.maxnum.f32(float %1008, float %928)
  %1013 = tail call float @llvm.maxnum.f32(float %1009, float %929)
  %1014 = tail call float @llvm.maxnum.f32(float %1010, float %930)
  %1015 = tail call float @llvm.maxnum.f32(float %1011, float %931)
  %1016 = tail call float @llvm.maxnum.f32(float %1012, float %932)
  %1017 = tail call float @llvm.maxnum.f32(float %1013, float %933)
  %1018 = tail call float @llvm.maxnum.f32(float %1014, float %934)
  %1019 = tail call float @llvm.maxnum.f32(float %1015, float %935)
  %1020 = tail call float @llvm.maxnum.f32(float %1016, float %936)
  %1021 = tail call float @llvm.maxnum.f32(float %1017, float %937)
  %1022 = tail call float @llvm.maxnum.f32(float %1018, float %938)
  %1023 = tail call float @llvm.maxnum.f32(float %1019, float %939)
  %1024 = tail call float @llvm.maxnum.f32(float %1020, float %940)
  %1025 = tail call float @llvm.maxnum.f32(float %1021, float %1022)
  %1026 = tail call float @llvm.maxnum.f32(float %1023, float %1024)
  %1027 = tail call float @llvm.maxnum.f32(float %1025, float %1026)
  %1028 = bitcast float %1027 to i32
  %1029 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %980, i32 %1028)
  %1030 = bitcast i32 %1029 to float
  %1031 = tail call float @llvm.maxnum.f32(float %1027, float %1030)
  %1032 = bitcast float %1031 to i32
  %1033 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %988, i32 %1032)
  %1034 = bitcast i32 %1033 to float
  %1035 = tail call float @llvm.maxnum.f32(float %1031, float %1034)
  %1036 = extractvalue [2 x <1 x float>] %214, 1
  %1037 = insertelement <1 x float> poison, float %1035, i64 0
  %1038 = tail call <1 x float> @llvm.maxnum.v1f32(<1 x float> %1037, <1 x float> %1036)
  %1039 = shufflevector <1 x float> %995, <1 x float> poison, <4 x i32> zeroinitializer
  %1040 = fsub <4 x float> %861, %1039
  %1041 = fsub <4 x float> %862, %1039
  %1042 = fsub <4 x float> %863, %1039
  %1043 = fsub <4 x float> %864, %1039
  %1044 = fsub <4 x float> %865, %1039
  %1045 = fsub <4 x float> %866, %1039
  %1046 = fsub <4 x float> %867, %1039
  %1047 = fsub <4 x float> %868, %1039
  %1048 = shufflevector <1 x float> %1038, <1 x float> poison, <4 x i32> zeroinitializer
  %1049 = fsub <4 x float> %869, %1048
  %1050 = fsub <4 x float> %870, %1048
  %1051 = fsub <4 x float> %871, %1048
  %1052 = fsub <4 x float> %872, %1048
  %1053 = fsub <4 x float> %873, %1048
  %1054 = fsub <4 x float> %874, %1048
  %1055 = fsub <4 x float> %875, %1048
  %1056 = fsub <4 x float> %876, %1048
  %1057 = fsub <1 x float> %993, %995
  %1058 = extractelement <1 x float> %1057, i64 0
  %1059 = tail call float @llvm.amdgcn.exp2.f32(float %1058)
  %1060 = insertelement <4 x float> poison, float %1059, i64 0
  %1061 = shufflevector <4 x float> %1060, <4 x float> poison, <4 x i32> zeroinitializer
  %1062 = extractvalue [2 x <1 x float>] %215, 0
  %1063 = extractelement <1 x float> %1062, i64 0
  %1064 = fmul float %1063, %1059
  %1065 = fsub <1 x float> %1036, %1038
  %1066 = extractelement <1 x float> %1065, i64 0
  %1067 = tail call float @llvm.amdgcn.exp2.f32(float %1066)
  %1068 = insertelement <4 x float> poison, float %1067, i64 0
  %1069 = shufflevector <4 x float> %1068, <4 x float> poison, <4 x i32> zeroinitializer
  %1070 = extractvalue [2 x <1 x float>] %215, 1
  %1071 = extractelement <1 x float> %1070, i64 0
  %1072 = fmul float %1071, %1067
  %1073 = extractelement <4 x float> %1040, i64 0
  %1074 = tail call float @llvm.amdgcn.exp2.f32(float %1073)
  %1075 = insertelement <4 x float> poison, float %1074, i64 0
  %1076 = extractelement <4 x float> %1040, i64 1
  %1077 = tail call float @llvm.amdgcn.exp2.f32(float %1076)
  %1078 = insertelement <4 x float> %1075, float %1077, i64 1
  %1079 = extractelement <4 x float> %1040, i64 2
  %1080 = tail call float @llvm.amdgcn.exp2.f32(float %1079)
  %1081 = insertelement <4 x float> %1078, float %1080, i64 2
  %1082 = extractelement <4 x float> %1040, i64 3
  %1083 = tail call float @llvm.amdgcn.exp2.f32(float %1082)
  %1084 = insertelement <4 x float> %1081, float %1083, i64 3
  %1085 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1084) #9, !frisk.register_lifetimes !3
  %1086 = fptrunc <4 x float> %1085 to <4 x half>
  %1087 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1086) #9, !frisk.register_lifetimes !3
  %1088 = extractelement <4 x float> %1041, i64 0
  %1089 = tail call float @llvm.amdgcn.exp2.f32(float %1088)
  %1090 = insertelement <4 x float> poison, float %1089, i64 0
  %1091 = extractelement <4 x float> %1041, i64 1
  %1092 = tail call float @llvm.amdgcn.exp2.f32(float %1091)
  %1093 = insertelement <4 x float> %1090, float %1092, i64 1
  %1094 = fadd float %1077, %1092
  %1095 = extractelement <4 x float> %1041, i64 2
  %1096 = tail call float @llvm.amdgcn.exp2.f32(float %1095)
  %1097 = insertelement <4 x float> %1093, float %1096, i64 2
  %1098 = fadd float %1080, %1096
  %1099 = extractelement <4 x float> %1041, i64 3
  %1100 = tail call float @llvm.amdgcn.exp2.f32(float %1099)
  %1101 = insertelement <4 x float> %1097, float %1100, i64 3
  %1102 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1101) #9, !frisk.register_lifetimes !3
  %1103 = fptrunc <4 x float> %1102 to <4 x half>
  %1104 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1103) #9, !frisk.register_lifetimes !3
  %1105 = fadd float %1083, %1100
  %1106 = extractelement <4 x float> %1042, i64 0
  %1107 = tail call float @llvm.amdgcn.exp2.f32(float %1106)
  %1108 = insertelement <4 x float> poison, float %1107, i64 0
  %1109 = extractelement <4 x float> %1042, i64 1
  %1110 = tail call float @llvm.amdgcn.exp2.f32(float %1109)
  %1111 = insertelement <4 x float> %1108, float %1110, i64 1
  %1112 = extractelement <4 x float> %1042, i64 2
  %1113 = tail call float @llvm.amdgcn.exp2.f32(float %1112)
  %1114 = insertelement <4 x float> %1111, float %1113, i64 2
  %1115 = fadd float %1113, %1098
  %1116 = extractelement <4 x float> %1042, i64 3
  %1117 = tail call float @llvm.amdgcn.exp2.f32(float %1116)
  %1118 = insertelement <4 x float> %1114, float %1117, i64 3
  %1119 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1118) #9, !frisk.register_lifetimes !3
  %1120 = fptrunc <4 x float> %1119 to <4 x half>
  %1121 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1120) #9, !frisk.register_lifetimes !3
  %1122 = fadd float %1117, %1105
  %1123 = extractelement <4 x float> %1043, i64 0
  %1124 = tail call float @llvm.amdgcn.exp2.f32(float %1123)
  %1125 = insertelement <4 x float> poison, float %1124, i64 0
  %1126 = extractelement <4 x float> %1043, i64 1
  %1127 = tail call float @llvm.amdgcn.exp2.f32(float %1126)
  %1128 = insertelement <4 x float> %1125, float %1127, i64 1
  %1129 = extractelement <4 x float> %1043, i64 2
  %1130 = tail call float @llvm.amdgcn.exp2.f32(float %1129)
  %1131 = insertelement <4 x float> %1128, float %1130, i64 2
  %1132 = extractelement <4 x float> %1043, i64 3
  %1133 = tail call float @llvm.amdgcn.exp2.f32(float %1132)
  %1134 = insertelement <4 x float> %1131, float %1133, i64 3
  %1135 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1134) #9, !frisk.register_lifetimes !3
  %1136 = fptrunc <4 x float> %1135 to <4 x half>
  %1137 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1136) #9, !frisk.register_lifetimes !3
  %1138 = fadd float %1133, %1122
  %1139 = extractelement <4 x float> %1044, i64 0
  %1140 = tail call float @llvm.amdgcn.exp2.f32(float %1139)
  %1141 = insertelement <4 x float> poison, float %1140, i64 0
  %1142 = extractelement <4 x float> %1044, i64 1
  %1143 = tail call float @llvm.amdgcn.exp2.f32(float %1142)
  %1144 = insertelement <4 x float> %1141, float %1143, i64 1
  %1145 = extractelement <4 x float> %1044, i64 2
  %1146 = tail call float @llvm.amdgcn.exp2.f32(float %1145)
  %1147 = insertelement <4 x float> %1144, float %1146, i64 2
  %1148 = extractelement <4 x float> %1044, i64 3
  %1149 = tail call float @llvm.amdgcn.exp2.f32(float %1148)
  %1150 = insertelement <4 x float> %1147, float %1149, i64 3
  %1151 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1150) #9, !frisk.register_lifetimes !3
  %1152 = fptrunc <4 x float> %1151 to <4 x half>
  %1153 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1152) #9, !frisk.register_lifetimes !3
  %1154 = fadd float %1149, %1138
  %1155 = extractelement <4 x float> %1045, i64 0
  %1156 = tail call float @llvm.amdgcn.exp2.f32(float %1155)
  %1157 = insertelement <4 x float> poison, float %1156, i64 0
  %1158 = extractelement <4 x float> %1045, i64 1
  %1159 = tail call float @llvm.amdgcn.exp2.f32(float %1158)
  %1160 = insertelement <4 x float> %1157, float %1159, i64 1
  %1161 = extractelement <4 x float> %1045, i64 2
  %1162 = tail call float @llvm.amdgcn.exp2.f32(float %1161)
  %1163 = insertelement <4 x float> %1160, float %1162, i64 2
  %1164 = extractelement <4 x float> %1045, i64 3
  %1165 = tail call float @llvm.amdgcn.exp2.f32(float %1164)
  %1166 = insertelement <4 x float> %1163, float %1165, i64 3
  %1167 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1166) #9, !frisk.register_lifetimes !3
  %1168 = fptrunc <4 x float> %1167 to <4 x half>
  %1169 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1168) #9, !frisk.register_lifetimes !3
  %1170 = fadd float %1165, %1154
  %1171 = extractelement <4 x float> %1046, i64 0
  %1172 = tail call float @llvm.amdgcn.exp2.f32(float %1171)
  %1173 = insertelement <4 x float> poison, float %1172, i64 0
  %1174 = extractelement <4 x float> %1046, i64 1
  %1175 = tail call float @llvm.amdgcn.exp2.f32(float %1174)
  %1176 = insertelement <4 x float> %1173, float %1175, i64 1
  %1177 = extractelement <4 x float> %1046, i64 2
  %1178 = tail call float @llvm.amdgcn.exp2.f32(float %1177)
  %1179 = insertelement <4 x float> %1176, float %1178, i64 2
  %1180 = extractelement <4 x float> %1046, i64 3
  %1181 = tail call float @llvm.amdgcn.exp2.f32(float %1180)
  %1182 = insertelement <4 x float> %1179, float %1181, i64 3
  %1183 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1182) #9, !frisk.register_lifetimes !3
  %1184 = fptrunc <4 x float> %1183 to <4 x half>
  %1185 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1184) #9, !frisk.register_lifetimes !3
  %1186 = fadd float %1181, %1170
  %1187 = extractelement <4 x float> %1047, i64 0
  %1188 = extractelement <4 x float> %1047, i64 1
  %1189 = tail call float @llvm.amdgcn.exp2.f32(float %1188)
  %1190 = extractelement <4 x float> %1047, i64 2
  %1191 = tail call float @llvm.amdgcn.exp2.f32(float %1190)
  %1192 = extractelement <4 x float> %1047, i64 3
  %1193 = tail call float @llvm.amdgcn.exp2.f32(float %1192)
  %1194 = fadd float %1193, %1186
  %1195 = insertelement <4 x float> <float -0.000000e+00, float poison, float poison, float poison>, float %1074, i64 1
  %1196 = insertelement <4 x float> %1195, float %1110, i64 2
  %1197 = insertelement <4 x float> %1196, float %1130, i64 3
  %1198 = insertelement <4 x float> <float -0.000000e+00, float poison, float poison, float poison>, float %1089, i64 1
  %1199 = insertelement <4 x float> %1198, float %1094, i64 2
  %1200 = insertelement <4 x float> %1199, float %1115, i64 3
  %1201 = fadd <4 x float> %1197, %1200
  %1202 = insertelement <4 x float> <float -0.000000e+00, float poison, float poison, float poison>, float %1107, i64 1
  %1203 = insertelement <4 x float> %1202, float %1127, i64 2
  %1204 = insertelement <4 x float> %1203, float %1146, i64 3
  %1205 = fadd <4 x float> %1204, %1201
  %1206 = insertelement <4 x float> <float -0.000000e+00, float poison, float poison, float poison>, float %1124, i64 1
  %1207 = insertelement <4 x float> %1206, float %1143, i64 2
  %1208 = insertelement <4 x float> %1207, float %1162, i64 3
  %1209 = fadd <4 x float> %1208, %1205
  %1210 = insertelement <4 x float> <float -0.000000e+00, float poison, float poison, float poison>, float %1140, i64 1
  %1211 = insertelement <4 x float> %1210, float %1159, i64 2
  %1212 = insertelement <4 x float> %1211, float %1178, i64 3
  %1213 = fadd <4 x float> %1212, %1209
  %1214 = tail call float @llvm.amdgcn.exp2.f32(float %1187)
  %1215 = insertelement <4 x float> poison, float %1214, i64 0
  %1216 = insertelement <4 x float> %1215, float %1156, i64 1
  %1217 = insertelement <4 x float> %1216, float %1175, i64 2
  %1218 = insertelement <4 x float> %1217, float %1191, i64 3
  %1219 = fadd <4 x float> %1218, %1213
  %1220 = insertelement <4 x float> <float -0.000000e+00, float poison, float poison, float poison>, float %1172, i64 1
  %1221 = insertelement <4 x float> %1220, float %1189, i64 2
  %1222 = insertelement <4 x float> %1221, float %1194, i64 3
  %1223 = fadd <4 x float> %1219, %1222
  %1224 = insertelement <4 x float> %1215, float %1189, i64 1
  %1225 = insertelement <4 x float> %1224, float %1191, i64 2
  %1226 = insertelement <4 x float> %1225, float %1193, i64 3
  %1227 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1226) #9, !frisk.register_lifetimes !3
  %1228 = fptrunc <4 x float> %1227 to <4 x half>
  %1229 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1228) #9, !frisk.register_lifetimes !3
  %1230 = tail call float @llvm.vector.reduce.fadd.v4f32(float -0.000000e+00, <4 x float> %1223)
  %1231 = extractelement <4 x float> %1049, i64 0
  %1232 = tail call float @llvm.amdgcn.exp2.f32(float %1231)
  %1233 = insertelement <4 x float> poison, float %1232, i64 0
  %1234 = extractelement <4 x float> %1049, i64 1
  %1235 = tail call float @llvm.amdgcn.exp2.f32(float %1234)
  %1236 = insertelement <4 x float> %1233, float %1235, i64 1
  %1237 = extractelement <4 x float> %1049, i64 2
  %1238 = tail call float @llvm.amdgcn.exp2.f32(float %1237)
  %1239 = insertelement <4 x float> %1236, float %1238, i64 2
  %1240 = extractelement <4 x float> %1049, i64 3
  %1241 = tail call float @llvm.amdgcn.exp2.f32(float %1240)
  %1242 = insertelement <4 x float> %1239, float %1241, i64 3
  %1243 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1242) #9, !frisk.register_lifetimes !3
  %1244 = fptrunc <4 x float> %1243 to <4 x half>
  %1245 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1244) #9, !frisk.register_lifetimes !3
  %1246 = extractelement <4 x float> %1050, i64 0
  %1247 = tail call float @llvm.amdgcn.exp2.f32(float %1246)
  %1248 = insertelement <4 x float> poison, float %1247, i64 0
  %1249 = extractelement <4 x float> %1050, i64 1
  %1250 = tail call float @llvm.amdgcn.exp2.f32(float %1249)
  %1251 = insertelement <4 x float> %1248, float %1250, i64 1
  %1252 = fadd float %1235, %1250
  %1253 = extractelement <4 x float> %1050, i64 2
  %1254 = tail call float @llvm.amdgcn.exp2.f32(float %1253)
  %1255 = insertelement <4 x float> %1251, float %1254, i64 2
  %1256 = fadd float %1238, %1254
  %1257 = extractelement <4 x float> %1050, i64 3
  %1258 = tail call float @llvm.amdgcn.exp2.f32(float %1257)
  %1259 = insertelement <4 x float> %1255, float %1258, i64 3
  %1260 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1259) #9, !frisk.register_lifetimes !3
  %1261 = fptrunc <4 x float> %1260 to <4 x half>
  %1262 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1261) #9, !frisk.register_lifetimes !3
  %1263 = fadd float %1241, %1258
  %1264 = extractelement <4 x float> %1051, i64 0
  %1265 = tail call float @llvm.amdgcn.exp2.f32(float %1264)
  %1266 = insertelement <4 x float> poison, float %1265, i64 0
  %1267 = extractelement <4 x float> %1051, i64 1
  %1268 = tail call float @llvm.amdgcn.exp2.f32(float %1267)
  %1269 = insertelement <4 x float> %1266, float %1268, i64 1
  %1270 = extractelement <4 x float> %1051, i64 2
  %1271 = tail call float @llvm.amdgcn.exp2.f32(float %1270)
  %1272 = insertelement <4 x float> %1269, float %1271, i64 2
  %1273 = fadd float %1271, %1256
  %1274 = extractelement <4 x float> %1051, i64 3
  %1275 = tail call float @llvm.amdgcn.exp2.f32(float %1274)
  %1276 = insertelement <4 x float> %1272, float %1275, i64 3
  %1277 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1276) #9, !frisk.register_lifetimes !3
  %1278 = fptrunc <4 x float> %1277 to <4 x half>
  %1279 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1278) #9, !frisk.register_lifetimes !3
  %1280 = fadd float %1275, %1263
  %1281 = extractelement <4 x float> %1052, i64 0
  %1282 = tail call float @llvm.amdgcn.exp2.f32(float %1281)
  %1283 = insertelement <4 x float> poison, float %1282, i64 0
  %1284 = extractelement <4 x float> %1052, i64 1
  %1285 = tail call float @llvm.amdgcn.exp2.f32(float %1284)
  %1286 = insertelement <4 x float> %1283, float %1285, i64 1
  %1287 = extractelement <4 x float> %1052, i64 2
  %1288 = tail call float @llvm.amdgcn.exp2.f32(float %1287)
  %1289 = insertelement <4 x float> %1286, float %1288, i64 2
  %1290 = extractelement <4 x float> %1052, i64 3
  %1291 = tail call float @llvm.amdgcn.exp2.f32(float %1290)
  %1292 = insertelement <4 x float> %1289, float %1291, i64 3
  %1293 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1292) #9, !frisk.register_lifetimes !3
  %1294 = fptrunc <4 x float> %1293 to <4 x half>
  %1295 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1294) #9, !frisk.register_lifetimes !3
  %1296 = fadd float %1291, %1280
  %1297 = extractelement <4 x float> %1053, i64 0
  %1298 = tail call float @llvm.amdgcn.exp2.f32(float %1297)
  %1299 = insertelement <4 x float> poison, float %1298, i64 0
  %1300 = extractelement <4 x float> %1053, i64 1
  %1301 = tail call float @llvm.amdgcn.exp2.f32(float %1300)
  %1302 = insertelement <4 x float> %1299, float %1301, i64 1
  %1303 = extractelement <4 x float> %1053, i64 2
  %1304 = tail call float @llvm.amdgcn.exp2.f32(float %1303)
  %1305 = insertelement <4 x float> %1302, float %1304, i64 2
  %1306 = extractelement <4 x float> %1053, i64 3
  %1307 = tail call float @llvm.amdgcn.exp2.f32(float %1306)
  %1308 = insertelement <4 x float> %1305, float %1307, i64 3
  %1309 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1308) #9, !frisk.register_lifetimes !3
  %1310 = fptrunc <4 x float> %1309 to <4 x half>
  %1311 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1310) #9, !frisk.register_lifetimes !3
  %1312 = fadd float %1307, %1296
  %1313 = extractelement <4 x float> %1054, i64 0
  %1314 = tail call float @llvm.amdgcn.exp2.f32(float %1313)
  %1315 = insertelement <4 x float> poison, float %1314, i64 0
  %1316 = extractelement <4 x float> %1054, i64 1
  %1317 = tail call float @llvm.amdgcn.exp2.f32(float %1316)
  %1318 = insertelement <4 x float> %1315, float %1317, i64 1
  %1319 = extractelement <4 x float> %1054, i64 2
  %1320 = tail call float @llvm.amdgcn.exp2.f32(float %1319)
  %1321 = insertelement <4 x float> %1318, float %1320, i64 2
  %1322 = extractelement <4 x float> %1054, i64 3
  %1323 = tail call float @llvm.amdgcn.exp2.f32(float %1322)
  %1324 = insertelement <4 x float> %1321, float %1323, i64 3
  %1325 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1324) #9, !frisk.register_lifetimes !3
  %1326 = fptrunc <4 x float> %1325 to <4 x half>
  %1327 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1326) #9, !frisk.register_lifetimes !3
  %1328 = fadd float %1323, %1312
  %1329 = extractelement <4 x float> %1055, i64 0
  %1330 = tail call float @llvm.amdgcn.exp2.f32(float %1329)
  %1331 = insertelement <4 x float> poison, float %1330, i64 0
  %1332 = extractelement <4 x float> %1055, i64 1
  %1333 = tail call float @llvm.amdgcn.exp2.f32(float %1332)
  %1334 = insertelement <4 x float> %1331, float %1333, i64 1
  %1335 = extractelement <4 x float> %1055, i64 2
  %1336 = tail call float @llvm.amdgcn.exp2.f32(float %1335)
  %1337 = insertelement <4 x float> %1334, float %1336, i64 2
  %1338 = extractelement <4 x float> %1055, i64 3
  %1339 = tail call float @llvm.amdgcn.exp2.f32(float %1338)
  %1340 = insertelement <4 x float> %1337, float %1339, i64 3
  %1341 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1340) #9, !frisk.register_lifetimes !3
  %1342 = fptrunc <4 x float> %1341 to <4 x half>
  %1343 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1342) #9, !frisk.register_lifetimes !3
  %1344 = fadd float %1339, %1328
  %1345 = extractelement <4 x float> %1056, i64 0
  %1346 = extractelement <4 x float> %1056, i64 1
  %1347 = tail call float @llvm.amdgcn.exp2.f32(float %1346)
  %1348 = extractelement <4 x float> %1056, i64 2
  %1349 = tail call float @llvm.amdgcn.exp2.f32(float %1348)
  %1350 = extractelement <4 x float> %1056, i64 3
  %1351 = tail call float @llvm.amdgcn.exp2.f32(float %1350)
  %1352 = fadd float %1351, %1344
  %1353 = insertelement <4 x float> <float -0.000000e+00, float poison, float poison, float poison>, float %1232, i64 1
  %1354 = insertelement <4 x float> %1353, float %1268, i64 2
  %1355 = insertelement <4 x float> %1354, float %1288, i64 3
  %1356 = insertelement <4 x float> <float -0.000000e+00, float poison, float poison, float poison>, float %1247, i64 1
  %1357 = insertelement <4 x float> %1356, float %1252, i64 2
  %1358 = insertelement <4 x float> %1357, float %1273, i64 3
  %1359 = fadd <4 x float> %1355, %1358
  %1360 = insertelement <4 x float> <float -0.000000e+00, float poison, float poison, float poison>, float %1265, i64 1
  %1361 = insertelement <4 x float> %1360, float %1285, i64 2
  %1362 = insertelement <4 x float> %1361, float %1304, i64 3
  %1363 = fadd <4 x float> %1362, %1359
  %1364 = insertelement <4 x float> <float -0.000000e+00, float poison, float poison, float poison>, float %1282, i64 1
  %1365 = insertelement <4 x float> %1364, float %1301, i64 2
  %1366 = insertelement <4 x float> %1365, float %1320, i64 3
  %1367 = fadd <4 x float> %1366, %1363
  %1368 = insertelement <4 x float> <float -0.000000e+00, float poison, float poison, float poison>, float %1298, i64 1
  %1369 = insertelement <4 x float> %1368, float %1317, i64 2
  %1370 = insertelement <4 x float> %1369, float %1336, i64 3
  %1371 = fadd <4 x float> %1370, %1367
  %1372 = tail call float @llvm.amdgcn.exp2.f32(float %1345)
  %1373 = insertelement <4 x float> poison, float %1372, i64 0
  %1374 = insertelement <4 x float> %1373, float %1314, i64 1
  %1375 = insertelement <4 x float> %1374, float %1333, i64 2
  %1376 = insertelement <4 x float> %1375, float %1349, i64 3
  %1377 = fadd <4 x float> %1376, %1371
  %1378 = insertelement <4 x float> <float -0.000000e+00, float poison, float poison, float poison>, float %1330, i64 1
  %1379 = insertelement <4 x float> %1378, float %1347, i64 2
  %1380 = insertelement <4 x float> %1379, float %1352, i64 3
  %1381 = fadd <4 x float> %1377, %1380
  %1382 = insertelement <4 x float> %1373, float %1347, i64 1
  %1383 = insertelement <4 x float> %1382, float %1349, i64 2
  %1384 = insertelement <4 x float> %1383, float %1351, i64 3
  %1385 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1384) #9, !frisk.register_lifetimes !3
  %1386 = fptrunc <4 x float> %1385 to <4 x half>
  %1387 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1386) #9, !frisk.register_lifetimes !3
  %1388 = tail call float @llvm.vector.reduce.fadd.v4f32(float -0.000000e+00, <4 x float> %1381)
  %1389 = bitcast float %1230 to i32
  %1390 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %980, i32 %1389)
  %1391 = bitcast i32 %1390 to float
  %1392 = fadd float %1230, %1391
  %1393 = bitcast float %1392 to i32
  %1394 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %988, i32 %1393)
  %1395 = bitcast i32 %1394 to float
  %1396 = fadd float %1392, %1395
  %1397 = fadd float %1064, %1396
  %1398 = insertelement <1 x float> poison, float %1397, i64 0
  %1399 = insertvalue [2 x <1 x float>] poison, <1 x float> %1398, 0
  %1400 = bitcast float %1388 to i32
  %1401 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %980, i32 %1400)
  %1402 = bitcast i32 %1401 to float
  %1403 = fadd float %1388, %1402
  %1404 = bitcast float %1403 to i32
  %1405 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %988, i32 %1404)
  %1406 = bitcast i32 %1405 to float
  %1407 = fadd float %1403, %1406
  %1408 = fadd float %1072, %1407
  %1409 = insertelement <1 x float> poison, float %1408, i64 0
  %1410 = insertvalue [2 x <1 x float>] %1399, <1 x float> %1409, 1
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %1411 = tail call i32 asm sideeffect "", "=v,0"(i32 %29) #9, !range !2, !frisk.register_lifetimes !3
  %1412 = shl nuw nsw i32 %1411, 3
  %1413 = and i32 %1412, 120
  %1414 = shl nuw nsw i32 %1411, 5
  %1415 = and i32 %1414, 1536
  %1416 = or disjoint i32 %1413, %1415
  %1417 = zext nneg i32 %1416 to i64
  %1418 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %1417
  %1419 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %1418, i32 0) #11
  %1420 = bitcast <4 x i32> %1419 to <8 x half>
  %1421 = zext nneg i32 %1411 to i64
  %1422 = and i64 %1421, 60
  %1423 = add nuw nsw i64 %1422, %1421
  %1424 = shl nuw nsw i64 %1423, 4
  %1425 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %1424
  %1426 = lshr i64 %1421, 4
  %1427 = and i64 %1426, 3
  %1428 = mul nuw nsw i64 %1427, 768
  %1429 = getelementptr i8, ptr addrspace(3) %1425, i64 %1428
  %1430 = lshr i64 %1421, 2
  %1431 = mul nsw i64 %1430, -64
  %1432 = getelementptr i8, ptr addrspace(3) %1429, i64 %1431
  %1433 = getelementptr i8, ptr addrspace(3) %1432, i64 256
  %1434 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %1433, i32 0) #11
  %1435 = bitcast <4 x i32> %1434 to <8 x half>
  %1436 = getelementptr i8, ptr addrspace(3) %1432, i64 512
  %1437 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %1436, i32 0) #11
  %1438 = bitcast <4 x i32> %1437 to <8 x half>
  %1439 = getelementptr i8, ptr addrspace(3) %1432, i64 768
  %1440 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %1439, i32 0) #11
  %1441 = bitcast <4 x i32> %1440 to <8 x half>
  %1442 = getelementptr i8, ptr addrspace(3) %1418, i64 4096
  %1443 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %1442, i32 0) #11
  %1444 = bitcast <4 x i32> %1443 to <8 x half>
  %1445 = getelementptr i8, ptr addrspace(3) %1432, i64 4352
  %1446 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %1445, i32 0) #11
  %1447 = bitcast <4 x i32> %1446 to <8 x half>
  %1448 = getelementptr i8, ptr addrspace(3) %1432, i64 4608
  %1449 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %1448, i32 0) #11
  %1450 = bitcast <4 x i32> %1449 to <8 x half>
  %1451 = getelementptr i8, ptr addrspace(3) %1432, i64 4864
  %1452 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %1451, i32 0) #11
  %1453 = bitcast <4 x i32> %1452 to <8 x half>
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %1454 = shl nuw nsw i64 %1421, 3
  %1455 = and i64 %1454, 1536
  %1456 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %1455
  %1457 = ptrtoint ptr addrspace(3) %1456 to i64
  %1458 = trunc i64 %1457 to i32
  %1459 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1458)
  %1460 = shl nuw nsw i64 %1421, 5
  %1461 = and i64 %1460, 8064
  %1462 = and i64 %1454, 24
  %1463 = or disjoint i64 %1461, %1462
  %1464 = or disjoint i64 %576, %1463
  %1465 = trunc i64 %1464 to i32
  %1466 = shl i32 %1465, 1
  %1467 = or i32 %1466, 64
  %1468 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1459)
  %1469 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %1468, i32 %1467, <4 x i32> %189) #10
  %1470 = or disjoint i64 %1421, 256
  %1471 = shl nuw nsw i64 %1470, 3
  %1472 = and i64 %1471, 3584
  %1473 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %1472
  %1474 = ptrtoint ptr addrspace(3) %1473 to i64
  %1475 = trunc i64 %1474 to i32
  %1476 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1475)
  %1477 = shl nuw nsw i64 %1470, 5
  %1478 = and i64 %1477, 16256
  %1479 = or disjoint i64 %1478, %1462
  %1480 = or disjoint i64 %576, %1479
  %1481 = trunc i64 %1480 to i32
  %1482 = shl i32 %1481, 1
  %1483 = or i32 %1482, 64
  %1484 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1476)
  %1485 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %1484, i32 %1483, <4 x i32> %189) #10
  %1486 = shufflevector <8 x half> %1420, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %1487 = extractvalue [2 x <8 x float>] %216, 0
  %1488 = shufflevector <8 x float> %1487, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %1489 = fmul <4 x float> %1488, %1061
  %1490 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1087, <4 x half> %1486, <4 x float> %1489) #9
  %1491 = shufflevector <8 x half> %1435, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %1492 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1104, <4 x half> %1491, <4 x float> %1490) #9
  %1493 = shufflevector <8 x half> %1438, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %1494 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1121, <4 x half> %1493, <4 x float> %1492) #9
  %1495 = shufflevector <8 x half> %1441, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %1496 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1137, <4 x half> %1495, <4 x float> %1494) #9
  %1497 = shufflevector <8 x half> %1444, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %1498 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1153, <4 x half> %1497, <4 x float> %1496) #9
  %1499 = shufflevector <8 x half> %1447, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %1500 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1169, <4 x half> %1499, <4 x float> %1498) #9
  %1501 = shufflevector <8 x half> %1450, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %1502 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1185, <4 x half> %1501, <4 x float> %1500) #9
  %1503 = shufflevector <8 x half> %1453, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %1504 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1229, <4 x half> %1503, <4 x float> %1502) #9
  %1505 = shufflevector <8 x half> %1420, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %1506 = shufflevector <8 x float> %1487, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %1507 = fmul <4 x float> %1506, %1061
  %1508 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1087, <4 x half> %1505, <4 x float> %1507) #9
  %1509 = shufflevector <8 x half> %1435, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %1510 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1104, <4 x half> %1509, <4 x float> %1508) #9
  %1511 = shufflevector <8 x half> %1438, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %1512 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1121, <4 x half> %1511, <4 x float> %1510) #9
  %1513 = shufflevector <8 x half> %1441, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %1514 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1137, <4 x half> %1513, <4 x float> %1512) #9
  %1515 = shufflevector <8 x half> %1444, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %1516 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1153, <4 x half> %1515, <4 x float> %1514) #9
  %1517 = shufflevector <8 x half> %1447, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %1518 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1169, <4 x half> %1517, <4 x float> %1516) #9
  %1519 = shufflevector <8 x half> %1450, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %1520 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1185, <4 x half> %1519, <4 x float> %1518) #9
  %1521 = shufflevector <8 x half> %1453, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %1522 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1229, <4 x half> %1521, <4 x float> %1520) #9
  %1523 = extractvalue [2 x <8 x float>] %216, 1
  %1524 = shufflevector <8 x float> %1523, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %1525 = fmul <4 x float> %1524, %1069
  %1526 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1245, <4 x half> %1486, <4 x float> %1525) #9
  %1527 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1262, <4 x half> %1491, <4 x float> %1526) #9
  %1528 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1279, <4 x half> %1493, <4 x float> %1527) #9
  %1529 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1295, <4 x half> %1495, <4 x float> %1528) #9
  %1530 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1311, <4 x half> %1497, <4 x float> %1529) #9
  %1531 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1327, <4 x half> %1499, <4 x float> %1530) #9
  %1532 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1343, <4 x half> %1501, <4 x float> %1531) #9
  %1533 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1387, <4 x half> %1503, <4 x float> %1532) #9
  %1534 = shufflevector <8 x float> %1523, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %1535 = fmul <4 x float> %1534, %1069
  %1536 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1245, <4 x half> %1505, <4 x float> %1535) #9
  %1537 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1262, <4 x half> %1509, <4 x float> %1536) #9
  %1538 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1279, <4 x half> %1511, <4 x float> %1537) #9
  %1539 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1295, <4 x half> %1513, <4 x float> %1538) #9
  %1540 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1311, <4 x half> %1515, <4 x float> %1539) #9
  %1541 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1327, <4 x half> %1517, <4 x float> %1540) #9
  %1542 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1343, <4 x half> %1519, <4 x float> %1541) #9
  %1543 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1387, <4 x half> %1521, <4 x float> %1542) #9
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %1544 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %1418, i32 0) #11
  %1545 = bitcast <4 x i32> %1544 to <8 x half>
  %1546 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %1433, i32 0) #11
  %1547 = bitcast <4 x i32> %1546 to <8 x half>
  %1548 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %1436, i32 0) #11
  %1549 = bitcast <4 x i32> %1548 to <8 x half>
  %1550 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %1439, i32 0) #11
  %1551 = bitcast <4 x i32> %1550 to <8 x half>
  %1552 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %1442, i32 0) #11
  %1553 = bitcast <4 x i32> %1552 to <8 x half>
  %1554 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %1445, i32 0) #11
  %1555 = bitcast <4 x i32> %1554 to <8 x half>
  %1556 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %1448, i32 0) #11
  %1557 = bitcast <4 x i32> %1556 to <8 x half>
  %1558 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %1451, i32 0) #11
  %1559 = bitcast <4 x i32> %1558 to <8 x half>
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %1560 = or i32 %1466, 128
  %1561 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %1468, i32 %1560, <4 x i32> %189) #10
  %1562 = or i32 %1482, 128
  %1563 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %1484, i32 %1562, <4 x i32> %189) #10
  %1564 = shufflevector <8 x half> %1545, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %1565 = extractvalue [2 x <8 x float>] %213, 0
  %1566 = shufflevector <8 x float> %1565, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %1567 = fmul <4 x float> %1566, %1061
  %1568 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1087, <4 x half> %1564, <4 x float> %1567) #9
  %1569 = shufflevector <8 x half> %1547, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %1570 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1104, <4 x half> %1569, <4 x float> %1568) #9
  %1571 = shufflevector <8 x half> %1549, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %1572 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1121, <4 x half> %1571, <4 x float> %1570) #9
  %1573 = shufflevector <8 x half> %1551, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %1574 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1137, <4 x half> %1573, <4 x float> %1572) #9
  %1575 = shufflevector <8 x half> %1553, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %1576 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1153, <4 x half> %1575, <4 x float> %1574) #9
  %1577 = shufflevector <8 x half> %1555, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %1578 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1169, <4 x half> %1577, <4 x float> %1576) #9
  %1579 = shufflevector <8 x half> %1557, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %1580 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1185, <4 x half> %1579, <4 x float> %1578) #9
  %1581 = shufflevector <8 x half> %1559, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %1582 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1229, <4 x half> %1581, <4 x float> %1580) #9
  %1583 = shufflevector <8 x half> %1545, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %1584 = shufflevector <8 x float> %1565, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %1585 = fmul <4 x float> %1584, %1061
  %1586 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1087, <4 x half> %1583, <4 x float> %1585) #9
  %1587 = shufflevector <8 x half> %1547, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %1588 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1104, <4 x half> %1587, <4 x float> %1586) #9
  %1589 = shufflevector <8 x half> %1549, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %1590 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1121, <4 x half> %1589, <4 x float> %1588) #9
  %1591 = shufflevector <8 x half> %1551, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %1592 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1137, <4 x half> %1591, <4 x float> %1590) #9
  %1593 = shufflevector <8 x half> %1553, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %1594 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1153, <4 x half> %1593, <4 x float> %1592) #9
  %1595 = shufflevector <8 x half> %1555, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %1596 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1169, <4 x half> %1595, <4 x float> %1594) #9
  %1597 = shufflevector <8 x half> %1557, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %1598 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1185, <4 x half> %1597, <4 x float> %1596) #9
  %1599 = shufflevector <8 x half> %1559, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %1600 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1229, <4 x half> %1599, <4 x float> %1598) #9
  %1601 = extractvalue [2 x <8 x float>] %213, 1
  %1602 = shufflevector <8 x float> %1601, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %1603 = fmul <4 x float> %1602, %1069
  %1604 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1245, <4 x half> %1564, <4 x float> %1603) #9
  %1605 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1262, <4 x half> %1569, <4 x float> %1604) #9
  %1606 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1279, <4 x half> %1571, <4 x float> %1605) #9
  %1607 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1295, <4 x half> %1573, <4 x float> %1606) #9
  %1608 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1311, <4 x half> %1575, <4 x float> %1607) #9
  %1609 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1327, <4 x half> %1577, <4 x float> %1608) #9
  %1610 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1343, <4 x half> %1579, <4 x float> %1609) #9
  %1611 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1387, <4 x half> %1581, <4 x float> %1610) #9
  %1612 = shufflevector <8 x float> %1601, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %1613 = fmul <4 x float> %1612, %1069
  %1614 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1245, <4 x half> %1583, <4 x float> %1613) #9
  %1615 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1262, <4 x half> %1587, <4 x float> %1614) #9
  %1616 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1279, <4 x half> %1589, <4 x float> %1615) #9
  %1617 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1295, <4 x half> %1591, <4 x float> %1616) #9
  %1618 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1311, <4 x half> %1593, <4 x float> %1617) #9
  %1619 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1327, <4 x half> %1595, <4 x float> %1618) #9
  %1620 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1343, <4 x half> %1597, <4 x float> %1619) #9
  %1621 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1387, <4 x half> %1599, <4 x float> %1620) #9
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %1622 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %1418, i32 0) #11
  %1623 = bitcast <4 x i32> %1622 to <8 x half>
  %1624 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %1433, i32 0) #11
  %1625 = bitcast <4 x i32> %1624 to <8 x half>
  %1626 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %1436, i32 0) #11
  %1627 = bitcast <4 x i32> %1626 to <8 x half>
  %1628 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %1439, i32 0) #11
  %1629 = bitcast <4 x i32> %1628 to <8 x half>
  %1630 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %1442, i32 0) #11
  %1631 = bitcast <4 x i32> %1630 to <8 x half>
  %1632 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %1445, i32 0) #11
  %1633 = bitcast <4 x i32> %1632 to <8 x half>
  %1634 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %1448, i32 0) #11
  %1635 = bitcast <4 x i32> %1634 to <8 x half>
  %1636 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %1451, i32 0) #11
  %1637 = bitcast <4 x i32> %1636 to <8 x half>
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %1638 = or disjoint i64 %576, 96
  %1639 = or disjoint i64 %1463, %1638
  %1640 = trunc i64 %1639 to i32
  %1641 = shl i32 %1640, 1
  %1642 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %1468, i32 %1641, <4 x i32> %189) #10
  %1643 = or disjoint i64 %1479, %1638
  %1644 = trunc i64 %1643 to i32
  %1645 = shl i32 %1644, 1
  %1646 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %1484, i32 %1645, <4 x i32> %189) #10
  %1647 = shufflevector <8 x half> %1623, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %1648 = extractvalue [2 x <8 x float>] %212, 0
  %1649 = shufflevector <8 x float> %1648, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %1650 = fmul <4 x float> %1649, %1061
  %1651 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1087, <4 x half> %1647, <4 x float> %1650) #9
  %1652 = shufflevector <8 x half> %1625, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %1653 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1104, <4 x half> %1652, <4 x float> %1651) #9
  %1654 = shufflevector <8 x half> %1627, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %1655 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1121, <4 x half> %1654, <4 x float> %1653) #9
  %1656 = shufflevector <8 x half> %1629, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %1657 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1137, <4 x half> %1656, <4 x float> %1655) #9
  %1658 = shufflevector <8 x half> %1631, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %1659 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1153, <4 x half> %1658, <4 x float> %1657) #9
  %1660 = shufflevector <8 x half> %1633, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %1661 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1169, <4 x half> %1660, <4 x float> %1659) #9
  %1662 = shufflevector <8 x half> %1635, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %1663 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1185, <4 x half> %1662, <4 x float> %1661) #9
  %1664 = shufflevector <8 x half> %1637, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %1665 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1229, <4 x half> %1664, <4 x float> %1663) #9
  %1666 = shufflevector <8 x half> %1623, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %1667 = shufflevector <8 x float> %1648, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %1668 = fmul <4 x float> %1667, %1061
  %1669 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1087, <4 x half> %1666, <4 x float> %1668) #9
  %1670 = shufflevector <8 x half> %1625, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %1671 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1104, <4 x half> %1670, <4 x float> %1669) #9
  %1672 = shufflevector <8 x half> %1627, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %1673 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1121, <4 x half> %1672, <4 x float> %1671) #9
  %1674 = shufflevector <8 x half> %1629, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %1675 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1137, <4 x half> %1674, <4 x float> %1673) #9
  %1676 = shufflevector <8 x half> %1631, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %1677 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1153, <4 x half> %1676, <4 x float> %1675) #9
  %1678 = shufflevector <8 x half> %1633, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %1679 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1169, <4 x half> %1678, <4 x float> %1677) #9
  %1680 = shufflevector <8 x half> %1635, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %1681 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1185, <4 x half> %1680, <4 x float> %1679) #9
  %1682 = shufflevector <8 x half> %1637, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %1683 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1229, <4 x half> %1682, <4 x float> %1681) #9
  %1684 = extractvalue [2 x <8 x float>] %212, 1
  %1685 = shufflevector <8 x float> %1684, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %1686 = fmul <4 x float> %1685, %1069
  %1687 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1245, <4 x half> %1647, <4 x float> %1686) #9
  %1688 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1262, <4 x half> %1652, <4 x float> %1687) #9
  %1689 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1279, <4 x half> %1654, <4 x float> %1688) #9
  %1690 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1295, <4 x half> %1656, <4 x float> %1689) #9
  %1691 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1311, <4 x half> %1658, <4 x float> %1690) #9
  %1692 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1327, <4 x half> %1660, <4 x float> %1691) #9
  %1693 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1343, <4 x half> %1662, <4 x float> %1692) #9
  %1694 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1387, <4 x half> %1664, <4 x float> %1693) #9
  %1695 = shufflevector <8 x float> %1684, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %1696 = fmul <4 x float> %1695, %1069
  %1697 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1245, <4 x half> %1666, <4 x float> %1696) #9
  %1698 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1262, <4 x half> %1670, <4 x float> %1697) #9
  %1699 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1279, <4 x half> %1672, <4 x float> %1698) #9
  %1700 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1295, <4 x half> %1674, <4 x float> %1699) #9
  %1701 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1311, <4 x half> %1676, <4 x float> %1700) #9
  %1702 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1327, <4 x half> %1678, <4 x float> %1701) #9
  %1703 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1343, <4 x half> %1680, <4 x float> %1702) #9
  %1704 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1387, <4 x half> %1682, <4 x float> %1703) #9
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %1705 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %1418, i32 0) #11
  %1706 = bitcast <4 x i32> %1705 to <8 x half>
  %1707 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %1433, i32 0) #11
  %1708 = bitcast <4 x i32> %1707 to <8 x half>
  %1709 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %1436, i32 0) #11
  %1710 = bitcast <4 x i32> %1709 to <8 x half>
  %1711 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %1439, i32 0) #11
  %1712 = bitcast <4 x i32> %1711 to <8 x half>
  %1713 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %1442, i32 0) #11
  %1714 = bitcast <4 x i32> %1713 to <8 x half>
  %1715 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %1445, i32 0) #11
  %1716 = bitcast <4 x i32> %1715 to <8 x half>
  %1717 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %1448, i32 0) #11
  %1718 = bitcast <4 x i32> %1717 to <8 x half>
  %1719 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %1451, i32 0) #11
  %1720 = bitcast <4 x i32> %1719 to <8 x half>
  %1721 = shufflevector <8 x half> %1706, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %1722 = extractvalue [2 x <8 x float>] %211, 0
  %1723 = shufflevector <8 x float> %1722, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %1724 = fmul <4 x float> %1723, %1061
  %1725 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1087, <4 x half> %1721, <4 x float> %1724) #9
  %1726 = shufflevector <8 x half> %1708, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %1727 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1104, <4 x half> %1726, <4 x float> %1725) #9
  %1728 = shufflevector <8 x half> %1710, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %1729 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1121, <4 x half> %1728, <4 x float> %1727) #9
  %1730 = shufflevector <8 x half> %1712, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %1731 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1137, <4 x half> %1730, <4 x float> %1729) #9
  %1732 = shufflevector <8 x half> %1714, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %1733 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1153, <4 x half> %1732, <4 x float> %1731) #9
  %1734 = shufflevector <8 x half> %1716, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %1735 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1169, <4 x half> %1734, <4 x float> %1733) #9
  %1736 = shufflevector <8 x half> %1718, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %1737 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1185, <4 x half> %1736, <4 x float> %1735) #9
  %1738 = shufflevector <8 x half> %1720, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %1739 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1229, <4 x half> %1738, <4 x float> %1737) #9
  %1740 = shufflevector <8 x half> %1706, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %1741 = shufflevector <8 x float> %1722, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %1742 = fmul <4 x float> %1741, %1061
  %1743 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1087, <4 x half> %1740, <4 x float> %1742) #9
  %1744 = shufflevector <8 x half> %1708, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %1745 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1104, <4 x half> %1744, <4 x float> %1743) #9
  %1746 = shufflevector <8 x half> %1710, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %1747 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1121, <4 x half> %1746, <4 x float> %1745) #9
  %1748 = shufflevector <8 x half> %1712, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %1749 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1137, <4 x half> %1748, <4 x float> %1747) #9
  %1750 = shufflevector <8 x half> %1714, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %1751 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1153, <4 x half> %1750, <4 x float> %1749) #9
  %1752 = shufflevector <8 x half> %1716, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %1753 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1169, <4 x half> %1752, <4 x float> %1751) #9
  %1754 = shufflevector <8 x half> %1718, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %1755 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1185, <4 x half> %1754, <4 x float> %1753) #9
  %1756 = shufflevector <8 x half> %1720, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %1757 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1229, <4 x half> %1756, <4 x float> %1755) #9
  %1758 = extractvalue [2 x <8 x float>] %211, 1
  %1759 = shufflevector <8 x float> %1758, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %1760 = fmul <4 x float> %1759, %1069
  %1761 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1245, <4 x half> %1721, <4 x float> %1760) #9
  %1762 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1262, <4 x half> %1726, <4 x float> %1761) #9
  %1763 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1279, <4 x half> %1728, <4 x float> %1762) #9
  %1764 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1295, <4 x half> %1730, <4 x float> %1763) #9
  %1765 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1311, <4 x half> %1732, <4 x float> %1764) #9
  %1766 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1327, <4 x half> %1734, <4 x float> %1765) #9
  %1767 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1343, <4 x half> %1736, <4 x float> %1766) #9
  %1768 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1387, <4 x half> %1738, <4 x float> %1767) #9
  %1769 = shufflevector <8 x float> %1758, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %1770 = fmul <4 x float> %1769, %1069
  %1771 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1245, <4 x half> %1740, <4 x float> %1770) #9
  %1772 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1262, <4 x half> %1744, <4 x float> %1771) #9
  %1773 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1279, <4 x half> %1746, <4 x float> %1772) #9
  %1774 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1295, <4 x half> %1748, <4 x float> %1773) #9
  %1775 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1311, <4 x half> %1750, <4 x float> %1774) #9
  %1776 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1327, <4 x half> %1752, <4 x float> %1775) #9
  %1777 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1343, <4 x half> %1754, <4 x float> %1776) #9
  %1778 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %1387, <4 x half> %1756, <4 x float> %1777) #9
  %1779 = shufflevector <4 x float> %1504, <4 x float> %1522, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1780 = insertvalue [2 x <8 x float>] poison, <8 x float> %1779, 0
  %1781 = shufflevector <4 x float> %1533, <4 x float> %1543, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1782 = insertvalue [2 x <8 x float>] %1780, <8 x float> %1781, 1
  %1783 = insertvalue [2 x <1 x float>] poison, <1 x float> %995, 0
  %1784 = insertvalue [2 x <1 x float>] %1783, <1 x float> %1038, 1
  %1785 = shufflevector <4 x float> %1582, <4 x float> %1600, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1786 = insertvalue [2 x <8 x float>] poison, <8 x float> %1785, 0
  %1787 = shufflevector <4 x float> %1611, <4 x float> %1621, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1788 = insertvalue [2 x <8 x float>] %1786, <8 x float> %1787, 1
  %1789 = shufflevector <4 x float> %1665, <4 x float> %1683, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1790 = insertvalue [2 x <8 x float>] poison, <8 x float> %1789, 0
  %1791 = shufflevector <4 x float> %1694, <4 x float> %1704, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1792 = insertvalue [2 x <8 x float>] %1790, <8 x float> %1791, 1
  %1793 = shufflevector <4 x float> %1739, <4 x float> %1757, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1794 = insertvalue [2 x <8 x float>] poison, <8 x float> %1793, 0
  %1795 = shufflevector <4 x float> %1768, <4 x float> %1778, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1796 = insertvalue [2 x <8 x float>] %1794, <8 x float> %1795, 1
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %1797 = add nuw nsw i64 %217, 1
  %1798 = icmp slt i64 %1797, %159
  br i1 %1798, label %210, label %1799

1799:                                             ; preds = %860, %173
  %1800 = phi i32 [ %179, %173 ], [ %985, %860 ]
  %1801 = phi i32 [ %178, %173 ], [ %977, %860 ]
  %1802 = phi i32 [ %177, %173 ], [ %976, %860 ]
  %1803 = phi i32 [ %175, %173 ], [ %974, %860 ]
  %1804 = phi [2 x <8 x float>] [ zeroinitializer, %173 ], [ %1782, %860 ]
  %1805 = phi [2 x <1 x float>] [ zeroinitializer, %173 ], [ %1410, %860 ]
  %1806 = phi [2 x <8 x float>] [ zeroinitializer, %173 ], [ %1788, %860 ]
  %1807 = phi [2 x <8 x float>] [ zeroinitializer, %173 ], [ %1792, %860 ]
  %1808 = phi [2 x <8 x float>] [ zeroinitializer, %173 ], [ %1796, %860 ]
  %1809 = and i32 %29, 16
  %1810 = icmp eq i32 %1809, 0
  %1811 = and i32 %29, 32
  %1812 = icmp eq i32 %1811, 0
  %1813 = extractvalue [2 x <8 x float>] %1808, 0
  %1814 = extractelement <8 x float> %1813, i64 0
  %1815 = extractelement <8 x float> %1813, i64 1
  %1816 = extractelement <8 x float> %1813, i64 2
  %1817 = extractelement <8 x float> %1813, i64 3
  %1818 = select i1 %1810, float %1815, float %1814
  %1819 = icmp samesign ult i32 %1802, %1801
  %1820 = select i1 %1819, i32 %1802, i32 %1803
  %1821 = shl nuw nsw i32 %1820, 2
  %1822 = bitcast float %1818 to i32
  %1823 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1821, i32 %1822)
  %1824 = bitcast i32 %1823 to float
  %1825 = select i1 %1810, float %1814, float %1824
  %1826 = select i1 %1810, float %1824, float %1815
  %1827 = select i1 %1810, float %1817, float %1816
  %1828 = bitcast float %1827 to i32
  %1829 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1821, i32 %1828)
  %1830 = bitcast i32 %1829 to float
  %1831 = select i1 %1810, float %1816, float %1830
  %1832 = select i1 %1810, float %1830, float %1817
  %1833 = select i1 %1812, float %1831, float %1825
  %1834 = icmp samesign ult i32 %1800, %1801
  %1835 = select i1 %1834, i32 %1800, i32 %1803
  %1836 = shl nuw nsw i32 %1835, 2
  %1837 = bitcast float %1833 to i32
  %1838 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1836, i32 %1837)
  %1839 = bitcast i32 %1838 to float
  %1840 = select i1 %1812, float %1832, float %1826
  %1841 = bitcast float %1840 to i32
  %1842 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1836, i32 %1841)
  %1843 = bitcast i32 %1842 to float
  %1844 = insertelement <4 x i1> poison, i1 %1812, i64 0
  %1845 = shufflevector <4 x i1> %1844, <4 x i1> poison, <4 x i32> zeroinitializer
  %1846 = insertelement <4 x float> poison, float %1825, i64 0
  %1847 = insertelement <4 x float> %1846, float %1826, i64 1
  %1848 = insertelement <4 x float> %1847, float %1839, i64 2
  %1849 = insertelement <4 x float> %1848, float %1843, i64 3
  %1850 = insertelement <4 x float> poison, float %1839, i64 0
  %1851 = insertelement <4 x float> %1850, float %1843, i64 1
  %1852 = insertelement <4 x float> %1851, float %1831, i64 2
  %1853 = insertelement <4 x float> %1852, float %1832, i64 3
  %1854 = select <4 x i1> %1845, <4 x float> %1849, <4 x float> %1853
  %1855 = extractvalue [2 x <8 x float>] %1807, 0
  %1856 = extractelement <8 x float> %1855, i64 0
  %1857 = extractelement <8 x float> %1855, i64 1
  %1858 = extractelement <8 x float> %1855, i64 2
  %1859 = extractelement <8 x float> %1855, i64 3
  %1860 = select i1 %1810, float %1857, float %1856
  %1861 = bitcast float %1860 to i32
  %1862 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1821, i32 %1861)
  %1863 = bitcast i32 %1862 to float
  %1864 = select i1 %1810, float %1856, float %1863
  %1865 = select i1 %1810, float %1863, float %1857
  %1866 = select i1 %1810, float %1859, float %1858
  %1867 = bitcast float %1866 to i32
  %1868 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1821, i32 %1867)
  %1869 = bitcast i32 %1868 to float
  %1870 = select i1 %1810, float %1858, float %1869
  %1871 = select i1 %1810, float %1869, float %1859
  %1872 = select i1 %1812, float %1870, float %1864
  %1873 = bitcast float %1872 to i32
  %1874 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1836, i32 %1873)
  %1875 = bitcast i32 %1874 to float
  %1876 = select i1 %1812, float %1871, float %1865
  %1877 = bitcast float %1876 to i32
  %1878 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1836, i32 %1877)
  %1879 = bitcast i32 %1878 to float
  %1880 = insertelement <4 x float> poison, float %1864, i64 0
  %1881 = insertelement <4 x float> %1880, float %1865, i64 1
  %1882 = insertelement <4 x float> %1881, float %1875, i64 2
  %1883 = insertelement <4 x float> %1882, float %1879, i64 3
  %1884 = insertelement <4 x float> poison, float %1875, i64 0
  %1885 = insertelement <4 x float> %1884, float %1879, i64 1
  %1886 = insertelement <4 x float> %1885, float %1870, i64 2
  %1887 = insertelement <4 x float> %1886, float %1871, i64 3
  %1888 = select <4 x i1> %1845, <4 x float> %1883, <4 x float> %1887
  %1889 = extractvalue [2 x <8 x float>] %1806, 0
  %1890 = extractelement <8 x float> %1889, i64 0
  %1891 = extractelement <8 x float> %1889, i64 1
  %1892 = extractelement <8 x float> %1889, i64 2
  %1893 = extractelement <8 x float> %1889, i64 3
  %1894 = select i1 %1810, float %1891, float %1890
  %1895 = bitcast float %1894 to i32
  %1896 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1821, i32 %1895)
  %1897 = bitcast i32 %1896 to float
  %1898 = select i1 %1810, float %1890, float %1897
  %1899 = select i1 %1810, float %1897, float %1891
  %1900 = select i1 %1810, float %1893, float %1892
  %1901 = bitcast float %1900 to i32
  %1902 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1821, i32 %1901)
  %1903 = bitcast i32 %1902 to float
  %1904 = select i1 %1810, float %1892, float %1903
  %1905 = select i1 %1810, float %1903, float %1893
  %1906 = select i1 %1812, float %1904, float %1898
  %1907 = bitcast float %1906 to i32
  %1908 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1836, i32 %1907)
  %1909 = bitcast i32 %1908 to float
  %1910 = select i1 %1812, float %1905, float %1899
  %1911 = bitcast float %1910 to i32
  %1912 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1836, i32 %1911)
  %1913 = bitcast i32 %1912 to float
  %1914 = insertelement <4 x float> poison, float %1898, i64 0
  %1915 = insertelement <4 x float> %1914, float %1899, i64 1
  %1916 = insertelement <4 x float> %1915, float %1909, i64 2
  %1917 = insertelement <4 x float> %1916, float %1913, i64 3
  %1918 = insertelement <4 x float> poison, float %1909, i64 0
  %1919 = insertelement <4 x float> %1918, float %1913, i64 1
  %1920 = insertelement <4 x float> %1919, float %1904, i64 2
  %1921 = insertelement <4 x float> %1920, float %1905, i64 3
  %1922 = select <4 x i1> %1845, <4 x float> %1917, <4 x float> %1921
  %1923 = extractvalue [2 x <8 x float>] %1804, 0
  %1924 = extractelement <8 x float> %1923, i64 0
  %1925 = extractelement <8 x float> %1923, i64 1
  %1926 = extractelement <8 x float> %1923, i64 2
  %1927 = extractelement <8 x float> %1923, i64 3
  %1928 = select i1 %1810, float %1925, float %1924
  %1929 = bitcast float %1928 to i32
  %1930 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1821, i32 %1929)
  %1931 = bitcast i32 %1930 to float
  %1932 = select i1 %1810, float %1924, float %1931
  %1933 = select i1 %1810, float %1931, float %1925
  %1934 = select i1 %1810, float %1927, float %1926
  %1935 = bitcast float %1934 to i32
  %1936 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1821, i32 %1935)
  %1937 = bitcast i32 %1936 to float
  %1938 = select i1 %1810, float %1926, float %1937
  %1939 = select i1 %1810, float %1937, float %1927
  %1940 = select i1 %1812, float %1938, float %1932
  %1941 = bitcast float %1940 to i32
  %1942 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1836, i32 %1941)
  %1943 = select i1 %1812, float %1939, float %1933
  %1944 = bitcast float %1943 to i32
  %1945 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1836, i32 %1944)
  %1946 = extractvalue [2 x <1 x float>] %1805, 0
  %1947 = shufflevector <1 x float> %1946, <1 x float> poison, <4 x i32> zeroinitializer
  %1948 = extractelement <8 x float> %1813, i64 4
  %1949 = extractelement <8 x float> %1813, i64 5
  %1950 = extractelement <8 x float> %1813, i64 6
  %1951 = extractelement <8 x float> %1813, i64 7
  %1952 = select i1 %1810, float %1949, float %1948
  %1953 = bitcast float %1952 to i32
  %1954 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1821, i32 %1953)
  %1955 = bitcast i32 %1954 to float
  %1956 = select i1 %1810, float %1948, float %1955
  %1957 = select i1 %1810, float %1955, float %1949
  %1958 = select i1 %1810, float %1951, float %1950
  %1959 = bitcast float %1958 to i32
  %1960 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1821, i32 %1959)
  %1961 = bitcast i32 %1960 to float
  %1962 = select i1 %1810, float %1950, float %1961
  %1963 = select i1 %1810, float %1961, float %1951
  %1964 = select i1 %1812, float %1962, float %1956
  %1965 = bitcast float %1964 to i32
  %1966 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1836, i32 %1965)
  %1967 = bitcast i32 %1966 to float
  %1968 = select i1 %1812, float %1963, float %1957
  %1969 = bitcast float %1968 to i32
  %1970 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1836, i32 %1969)
  %1971 = bitcast i32 %1970 to float
  %1972 = insertelement <4 x float> poison, float %1956, i64 0
  %1973 = insertelement <4 x float> %1972, float %1957, i64 1
  %1974 = insertelement <4 x float> %1973, float %1967, i64 2
  %1975 = insertelement <4 x float> %1974, float %1971, i64 3
  %1976 = insertelement <4 x float> poison, float %1967, i64 0
  %1977 = insertelement <4 x float> %1976, float %1971, i64 1
  %1978 = insertelement <4 x float> %1977, float %1962, i64 2
  %1979 = insertelement <4 x float> %1978, float %1963, i64 3
  %1980 = select <4 x i1> %1845, <4 x float> %1975, <4 x float> %1979
  %1981 = extractelement <8 x float> %1855, i64 4
  %1982 = extractelement <8 x float> %1855, i64 5
  %1983 = extractelement <8 x float> %1855, i64 6
  %1984 = extractelement <8 x float> %1855, i64 7
  %1985 = select i1 %1810, float %1982, float %1981
  %1986 = bitcast float %1985 to i32
  %1987 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1821, i32 %1986)
  %1988 = bitcast i32 %1987 to float
  %1989 = select i1 %1810, float %1981, float %1988
  %1990 = select i1 %1810, float %1988, float %1982
  %1991 = select i1 %1810, float %1984, float %1983
  %1992 = bitcast float %1991 to i32
  %1993 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1821, i32 %1992)
  %1994 = bitcast i32 %1993 to float
  %1995 = select i1 %1810, float %1983, float %1994
  %1996 = select i1 %1810, float %1994, float %1984
  %1997 = select i1 %1812, float %1995, float %1989
  %1998 = bitcast float %1997 to i32
  %1999 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1836, i32 %1998)
  %2000 = bitcast i32 %1999 to float
  %2001 = select i1 %1812, float %1996, float %1990
  %2002 = bitcast float %2001 to i32
  %2003 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1836, i32 %2002)
  %2004 = bitcast i32 %2003 to float
  %2005 = insertelement <4 x float> poison, float %1989, i64 0
  %2006 = insertelement <4 x float> %2005, float %1990, i64 1
  %2007 = insertelement <4 x float> %2006, float %2000, i64 2
  %2008 = insertelement <4 x float> %2007, float %2004, i64 3
  %2009 = insertelement <4 x float> poison, float %2000, i64 0
  %2010 = insertelement <4 x float> %2009, float %2004, i64 1
  %2011 = insertelement <4 x float> %2010, float %1995, i64 2
  %2012 = insertelement <4 x float> %2011, float %1996, i64 3
  %2013 = select <4 x i1> %1845, <4 x float> %2008, <4 x float> %2012
  %2014 = extractelement <8 x float> %1889, i64 4
  %2015 = extractelement <8 x float> %1889, i64 5
  %2016 = extractelement <8 x float> %1889, i64 6
  %2017 = extractelement <8 x float> %1889, i64 7
  %2018 = select i1 %1810, float %2015, float %2014
  %2019 = bitcast float %2018 to i32
  %2020 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1821, i32 %2019)
  %2021 = bitcast i32 %2020 to float
  %2022 = select i1 %1810, float %2014, float %2021
  %2023 = select i1 %1810, float %2021, float %2015
  %2024 = select i1 %1810, float %2017, float %2016
  %2025 = bitcast float %2024 to i32
  %2026 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1821, i32 %2025)
  %2027 = bitcast i32 %2026 to float
  %2028 = select i1 %1810, float %2016, float %2027
  %2029 = select i1 %1810, float %2027, float %2017
  %2030 = select i1 %1812, float %2028, float %2022
  %2031 = bitcast float %2030 to i32
  %2032 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1836, i32 %2031)
  %2033 = bitcast i32 %2032 to float
  %2034 = select i1 %1812, float %2029, float %2023
  %2035 = bitcast float %2034 to i32
  %2036 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1836, i32 %2035)
  %2037 = bitcast i32 %2036 to float
  %2038 = insertelement <4 x float> poison, float %2022, i64 0
  %2039 = insertelement <4 x float> %2038, float %2023, i64 1
  %2040 = insertelement <4 x float> %2039, float %2033, i64 2
  %2041 = insertelement <4 x float> %2040, float %2037, i64 3
  %2042 = insertelement <4 x float> poison, float %2033, i64 0
  %2043 = insertelement <4 x float> %2042, float %2037, i64 1
  %2044 = insertelement <4 x float> %2043, float %2028, i64 2
  %2045 = insertelement <4 x float> %2044, float %2029, i64 3
  %2046 = select <4 x i1> %1845, <4 x float> %2041, <4 x float> %2045
  %2047 = extractelement <8 x float> %1923, i64 4
  %2048 = extractelement <8 x float> %1923, i64 5
  %2049 = extractelement <8 x float> %1923, i64 6
  %2050 = extractelement <8 x float> %1923, i64 7
  %2051 = select i1 %1810, float %2048, float %2047
  %2052 = bitcast float %2051 to i32
  %2053 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1821, i32 %2052)
  %2054 = bitcast i32 %2053 to float
  %2055 = select i1 %1810, float %2047, float %2054
  %2056 = select i1 %1810, float %2054, float %2048
  %2057 = select i1 %1810, float %2050, float %2049
  %2058 = bitcast float %2057 to i32
  %2059 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1821, i32 %2058)
  %2060 = bitcast i32 %2059 to float
  %2061 = select i1 %1810, float %2049, float %2060
  %2062 = select i1 %1810, float %2060, float %2050
  %2063 = select i1 %1812, float %2061, float %2055
  %2064 = bitcast float %2063 to i32
  %2065 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1836, i32 %2064)
  %2066 = select i1 %1812, float %2062, float %2056
  %2067 = bitcast float %2066 to i32
  %2068 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1836, i32 %2067)
  %2069 = extractvalue [2 x <8 x float>] %1808, 1
  %2070 = extractelement <8 x float> %2069, i64 0
  %2071 = extractelement <8 x float> %2069, i64 1
  %2072 = extractelement <8 x float> %2069, i64 2
  %2073 = extractelement <8 x float> %2069, i64 3
  %2074 = select i1 %1810, float %2071, float %2070
  %2075 = bitcast float %2074 to i32
  %2076 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1821, i32 %2075)
  %2077 = bitcast i32 %2076 to float
  %2078 = select i1 %1810, float %2070, float %2077
  %2079 = select i1 %1810, float %2077, float %2071
  %2080 = select i1 %1810, float %2073, float %2072
  %2081 = bitcast float %2080 to i32
  %2082 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1821, i32 %2081)
  %2083 = bitcast i32 %2082 to float
  %2084 = select i1 %1810, float %2072, float %2083
  %2085 = select i1 %1810, float %2083, float %2073
  %2086 = select i1 %1812, float %2084, float %2078
  %2087 = bitcast float %2086 to i32
  %2088 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1836, i32 %2087)
  %2089 = bitcast i32 %2088 to float
  %2090 = select i1 %1812, float %2085, float %2079
  %2091 = bitcast float %2090 to i32
  %2092 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1836, i32 %2091)
  %2093 = bitcast i32 %2092 to float
  %2094 = insertelement <4 x float> poison, float %2078, i64 0
  %2095 = insertelement <4 x float> %2094, float %2079, i64 1
  %2096 = insertelement <4 x float> %2095, float %2089, i64 2
  %2097 = insertelement <4 x float> %2096, float %2093, i64 3
  %2098 = insertelement <4 x float> poison, float %2089, i64 0
  %2099 = insertelement <4 x float> %2098, float %2093, i64 1
  %2100 = insertelement <4 x float> %2099, float %2084, i64 2
  %2101 = insertelement <4 x float> %2100, float %2085, i64 3
  %2102 = select <4 x i1> %1845, <4 x float> %2097, <4 x float> %2101
  %2103 = extractvalue [2 x <8 x float>] %1807, 1
  %2104 = extractelement <8 x float> %2103, i64 0
  %2105 = extractelement <8 x float> %2103, i64 1
  %2106 = extractelement <8 x float> %2103, i64 2
  %2107 = extractelement <8 x float> %2103, i64 3
  %2108 = select i1 %1810, float %2105, float %2104
  %2109 = bitcast float %2108 to i32
  %2110 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1821, i32 %2109)
  %2111 = bitcast i32 %2110 to float
  %2112 = select i1 %1810, float %2104, float %2111
  %2113 = select i1 %1810, float %2111, float %2105
  %2114 = select i1 %1810, float %2107, float %2106
  %2115 = bitcast float %2114 to i32
  %2116 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1821, i32 %2115)
  %2117 = bitcast i32 %2116 to float
  %2118 = select i1 %1810, float %2106, float %2117
  %2119 = select i1 %1810, float %2117, float %2107
  %2120 = select i1 %1812, float %2118, float %2112
  %2121 = bitcast float %2120 to i32
  %2122 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1836, i32 %2121)
  %2123 = bitcast i32 %2122 to float
  %2124 = select i1 %1812, float %2119, float %2113
  %2125 = bitcast float %2124 to i32
  %2126 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1836, i32 %2125)
  %2127 = bitcast i32 %2126 to float
  %2128 = insertelement <4 x float> poison, float %2112, i64 0
  %2129 = insertelement <4 x float> %2128, float %2113, i64 1
  %2130 = insertelement <4 x float> %2129, float %2123, i64 2
  %2131 = insertelement <4 x float> %2130, float %2127, i64 3
  %2132 = insertelement <4 x float> poison, float %2123, i64 0
  %2133 = insertelement <4 x float> %2132, float %2127, i64 1
  %2134 = insertelement <4 x float> %2133, float %2118, i64 2
  %2135 = insertelement <4 x float> %2134, float %2119, i64 3
  %2136 = select <4 x i1> %1845, <4 x float> %2131, <4 x float> %2135
  %2137 = extractvalue [2 x <8 x float>] %1806, 1
  %2138 = extractelement <8 x float> %2137, i64 0
  %2139 = extractelement <8 x float> %2137, i64 1
  %2140 = extractelement <8 x float> %2137, i64 2
  %2141 = extractelement <8 x float> %2137, i64 3
  %2142 = select i1 %1810, float %2139, float %2138
  %2143 = bitcast float %2142 to i32
  %2144 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1821, i32 %2143)
  %2145 = bitcast i32 %2144 to float
  %2146 = select i1 %1810, float %2138, float %2145
  %2147 = select i1 %1810, float %2145, float %2139
  %2148 = select i1 %1810, float %2141, float %2140
  %2149 = bitcast float %2148 to i32
  %2150 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1821, i32 %2149)
  %2151 = bitcast i32 %2150 to float
  %2152 = select i1 %1810, float %2140, float %2151
  %2153 = select i1 %1810, float %2151, float %2141
  %2154 = select i1 %1812, float %2152, float %2146
  %2155 = bitcast float %2154 to i32
  %2156 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1836, i32 %2155)
  %2157 = bitcast i32 %2156 to float
  %2158 = select i1 %1812, float %2153, float %2147
  %2159 = bitcast float %2158 to i32
  %2160 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1836, i32 %2159)
  %2161 = bitcast i32 %2160 to float
  %2162 = insertelement <4 x float> poison, float %2146, i64 0
  %2163 = insertelement <4 x float> %2162, float %2147, i64 1
  %2164 = insertelement <4 x float> %2163, float %2157, i64 2
  %2165 = insertelement <4 x float> %2164, float %2161, i64 3
  %2166 = insertelement <4 x float> poison, float %2157, i64 0
  %2167 = insertelement <4 x float> %2166, float %2161, i64 1
  %2168 = insertelement <4 x float> %2167, float %2152, i64 2
  %2169 = insertelement <4 x float> %2168, float %2153, i64 3
  %2170 = select <4 x i1> %1845, <4 x float> %2165, <4 x float> %2169
  %2171 = extractvalue [2 x <8 x float>] %1804, 1
  %2172 = extractelement <8 x float> %2171, i64 0
  %2173 = extractelement <8 x float> %2171, i64 1
  %2174 = extractelement <8 x float> %2171, i64 2
  %2175 = extractelement <8 x float> %2171, i64 3
  %2176 = select i1 %1810, float %2173, float %2172
  %2177 = bitcast float %2176 to i32
  %2178 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1821, i32 %2177)
  %2179 = bitcast i32 %2178 to float
  %2180 = select i1 %1810, float %2172, float %2179
  %2181 = select i1 %1810, float %2179, float %2173
  %2182 = select i1 %1810, float %2175, float %2174
  %2183 = bitcast float %2182 to i32
  %2184 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1821, i32 %2183)
  %2185 = bitcast i32 %2184 to float
  %2186 = select i1 %1810, float %2174, float %2185
  %2187 = select i1 %1810, float %2185, float %2175
  %2188 = select i1 %1812, float %2186, float %2180
  %2189 = bitcast float %2188 to i32
  %2190 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1836, i32 %2189)
  %2191 = select i1 %1812, float %2187, float %2181
  %2192 = bitcast float %2191 to i32
  %2193 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1836, i32 %2192)
  %2194 = extractvalue [2 x <1 x float>] %1805, 1
  %2195 = shufflevector <1 x float> %2194, <1 x float> poison, <4 x i32> zeroinitializer
  %2196 = extractelement <8 x float> %2069, i64 4
  %2197 = extractelement <8 x float> %2069, i64 5
  %2198 = extractelement <8 x float> %2069, i64 6
  %2199 = extractelement <8 x float> %2069, i64 7
  %2200 = select i1 %1810, float %2197, float %2196
  %2201 = bitcast float %2200 to i32
  %2202 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1821, i32 %2201)
  %2203 = bitcast i32 %2202 to float
  %2204 = select i1 %1810, float %2196, float %2203
  %2205 = select i1 %1810, float %2203, float %2197
  %2206 = select i1 %1810, float %2199, float %2198
  %2207 = bitcast float %2206 to i32
  %2208 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1821, i32 %2207)
  %2209 = bitcast i32 %2208 to float
  %2210 = select i1 %1810, float %2198, float %2209
  %2211 = select i1 %1810, float %2209, float %2199
  %2212 = select i1 %1812, float %2210, float %2204
  %2213 = bitcast float %2212 to i32
  %2214 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1836, i32 %2213)
  %2215 = bitcast i32 %2214 to float
  %2216 = select i1 %1812, float %2211, float %2205
  %2217 = bitcast float %2216 to i32
  %2218 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1836, i32 %2217)
  %2219 = bitcast i32 %2218 to float
  %2220 = insertelement <4 x float> poison, float %2204, i64 0
  %2221 = insertelement <4 x float> %2220, float %2205, i64 1
  %2222 = insertelement <4 x float> %2221, float %2215, i64 2
  %2223 = insertelement <4 x float> %2222, float %2219, i64 3
  %2224 = insertelement <4 x float> poison, float %2215, i64 0
  %2225 = insertelement <4 x float> %2224, float %2219, i64 1
  %2226 = insertelement <4 x float> %2225, float %2210, i64 2
  %2227 = insertelement <4 x float> %2226, float %2211, i64 3
  %2228 = select <4 x i1> %1845, <4 x float> %2223, <4 x float> %2227
  %2229 = extractelement <8 x float> %2103, i64 4
  %2230 = extractelement <8 x float> %2103, i64 5
  %2231 = extractelement <8 x float> %2103, i64 6
  %2232 = extractelement <8 x float> %2103, i64 7
  %2233 = select i1 %1810, float %2230, float %2229
  %2234 = bitcast float %2233 to i32
  %2235 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1821, i32 %2234)
  %2236 = bitcast i32 %2235 to float
  %2237 = select i1 %1810, float %2229, float %2236
  %2238 = select i1 %1810, float %2236, float %2230
  %2239 = select i1 %1810, float %2232, float %2231
  %2240 = bitcast float %2239 to i32
  %2241 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1821, i32 %2240)
  %2242 = bitcast i32 %2241 to float
  %2243 = select i1 %1810, float %2231, float %2242
  %2244 = select i1 %1810, float %2242, float %2232
  %2245 = select i1 %1812, float %2243, float %2237
  %2246 = bitcast float %2245 to i32
  %2247 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1836, i32 %2246)
  %2248 = bitcast i32 %2247 to float
  %2249 = select i1 %1812, float %2244, float %2238
  %2250 = bitcast float %2249 to i32
  %2251 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1836, i32 %2250)
  %2252 = bitcast i32 %2251 to float
  %2253 = insertelement <4 x float> poison, float %2237, i64 0
  %2254 = insertelement <4 x float> %2253, float %2238, i64 1
  %2255 = insertelement <4 x float> %2254, float %2248, i64 2
  %2256 = insertelement <4 x float> %2255, float %2252, i64 3
  %2257 = insertelement <4 x float> poison, float %2248, i64 0
  %2258 = insertelement <4 x float> %2257, float %2252, i64 1
  %2259 = insertelement <4 x float> %2258, float %2243, i64 2
  %2260 = insertelement <4 x float> %2259, float %2244, i64 3
  %2261 = select <4 x i1> %1845, <4 x float> %2256, <4 x float> %2260
  %2262 = extractelement <8 x float> %2137, i64 4
  %2263 = extractelement <8 x float> %2137, i64 5
  %2264 = extractelement <8 x float> %2137, i64 6
  %2265 = extractelement <8 x float> %2137, i64 7
  %2266 = select i1 %1810, float %2263, float %2262
  %2267 = bitcast float %2266 to i32
  %2268 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1821, i32 %2267)
  %2269 = bitcast i32 %2268 to float
  %2270 = select i1 %1810, float %2262, float %2269
  %2271 = select i1 %1810, float %2269, float %2263
  %2272 = select i1 %1810, float %2265, float %2264
  %2273 = bitcast float %2272 to i32
  %2274 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1821, i32 %2273)
  %2275 = bitcast i32 %2274 to float
  %2276 = select i1 %1810, float %2264, float %2275
  %2277 = select i1 %1810, float %2275, float %2265
  %2278 = select i1 %1812, float %2276, float %2270
  %2279 = bitcast float %2278 to i32
  %2280 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1836, i32 %2279)
  %2281 = bitcast i32 %2280 to float
  %2282 = select i1 %1812, float %2277, float %2271
  %2283 = bitcast float %2282 to i32
  %2284 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1836, i32 %2283)
  %2285 = bitcast i32 %2284 to float
  %2286 = insertelement <4 x float> poison, float %2270, i64 0
  %2287 = insertelement <4 x float> %2286, float %2271, i64 1
  %2288 = insertelement <4 x float> %2287, float %2281, i64 2
  %2289 = insertelement <4 x float> %2288, float %2285, i64 3
  %2290 = insertelement <4 x float> poison, float %2281, i64 0
  %2291 = insertelement <4 x float> %2290, float %2285, i64 1
  %2292 = insertelement <4 x float> %2291, float %2276, i64 2
  %2293 = insertelement <4 x float> %2292, float %2277, i64 3
  %2294 = select <4 x i1> %1845, <4 x float> %2289, <4 x float> %2293
  %2295 = extractelement <8 x float> %2171, i64 4
  %2296 = extractelement <8 x float> %2171, i64 5
  %2297 = extractelement <8 x float> %2171, i64 6
  %2298 = extractelement <8 x float> %2171, i64 7
  %2299 = select i1 %1810, float %2296, float %2295
  %2300 = bitcast float %2299 to i32
  %2301 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1821, i32 %2300)
  %2302 = bitcast i32 %2301 to float
  %2303 = select i1 %1810, float %2295, float %2302
  %2304 = select i1 %1810, float %2302, float %2296
  %2305 = select i1 %1810, float %2298, float %2297
  %2306 = bitcast float %2305 to i32
  %2307 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1821, i32 %2306)
  %2308 = bitcast i32 %2307 to float
  %2309 = select i1 %1810, float %2297, float %2308
  %2310 = select i1 %1810, float %2308, float %2298
  %2311 = select i1 %1812, float %2309, float %2303
  %2312 = bitcast float %2311 to i32
  %2313 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1836, i32 %2312)
  %2314 = select i1 %1812, float %2310, float %2304
  %2315 = bitcast float %2314 to i32
  %2316 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1836, i32 %2315)
  %2317 = bitcast i32 %2313 to float
  %2318 = bitcast i32 %2316 to float
  %2319 = insertelement <4 x float> poison, float %2303, i64 0
  %2320 = insertelement <4 x float> %2319, float %2304, i64 1
  %2321 = insertelement <4 x float> %2320, float %2317, i64 2
  %2322 = insertelement <4 x float> %2321, float %2318, i64 3
  %2323 = insertelement <4 x float> poison, float %2317, i64 0
  %2324 = insertelement <4 x float> %2323, float %2318, i64 1
  %2325 = insertelement <4 x float> %2324, float %2309, i64 2
  %2326 = insertelement <4 x float> %2325, float %2310, i64 3
  %2327 = select <4 x i1> %1845, <4 x float> %2322, <4 x float> %2326
  %2328 = fdiv arcp <4 x float> splat (float 1.000000e+00), %2195
  %2329 = fmul <4 x float> %2328, %2327
  %2330 = fptrunc <4 x float> %2329 to <4 x half>
  %2331 = extractelement <4 x half> %2330, i64 3
  %2332 = extractelement <4 x half> %2330, i64 2
  %2333 = extractelement <4 x half> %2330, i64 1
  %2334 = extractelement <4 x half> %2330, i64 0
  %2335 = bitcast i32 %2190 to float
  %2336 = bitcast i32 %2193 to float
  %2337 = insertelement <4 x float> poison, float %2180, i64 0
  %2338 = insertelement <4 x float> %2337, float %2181, i64 1
  %2339 = insertelement <4 x float> %2338, float %2335, i64 2
  %2340 = insertelement <4 x float> %2339, float %2336, i64 3
  %2341 = insertelement <4 x float> poison, float %2335, i64 0
  %2342 = insertelement <4 x float> %2341, float %2336, i64 1
  %2343 = insertelement <4 x float> %2342, float %2186, i64 2
  %2344 = insertelement <4 x float> %2343, float %2187, i64 3
  %2345 = select <4 x i1> %1845, <4 x float> %2340, <4 x float> %2344
  %2346 = fmul <4 x float> %2328, %2345
  %2347 = fptrunc <4 x float> %2346 to <4 x half>
  %2348 = extractelement <4 x half> %2347, i64 3
  %2349 = extractelement <4 x half> %2347, i64 2
  %2350 = extractelement <4 x half> %2347, i64 1
  %2351 = extractelement <4 x half> %2347, i64 0
  %2352 = bitcast i32 %2065 to float
  %2353 = bitcast i32 %2068 to float
  %2354 = insertelement <4 x float> poison, float %2055, i64 0
  %2355 = insertelement <4 x float> %2354, float %2056, i64 1
  %2356 = insertelement <4 x float> %2355, float %2352, i64 2
  %2357 = insertelement <4 x float> %2356, float %2353, i64 3
  %2358 = insertelement <4 x float> poison, float %2352, i64 0
  %2359 = insertelement <4 x float> %2358, float %2353, i64 1
  %2360 = insertelement <4 x float> %2359, float %2061, i64 2
  %2361 = insertelement <4 x float> %2360, float %2062, i64 3
  %2362 = select <4 x i1> %1845, <4 x float> %2357, <4 x float> %2361
  %2363 = fdiv arcp <4 x float> splat (float 1.000000e+00), %1947
  %2364 = fmul <4 x float> %2363, %2362
  %2365 = fptrunc <4 x float> %2364 to <4 x half>
  %2366 = extractelement <4 x half> %2365, i64 3
  %2367 = extractelement <4 x half> %2365, i64 2
  %2368 = extractelement <4 x half> %2365, i64 1
  %2369 = extractelement <4 x half> %2365, i64 0
  %2370 = bitcast i32 %1942 to float
  %2371 = bitcast i32 %1945 to float
  %2372 = insertelement <4 x float> poison, float %1932, i64 0
  %2373 = insertelement <4 x float> %2372, float %1933, i64 1
  %2374 = insertelement <4 x float> %2373, float %2370, i64 2
  %2375 = insertelement <4 x float> %2374, float %2371, i64 3
  %2376 = insertelement <4 x float> poison, float %2370, i64 0
  %2377 = insertelement <4 x float> %2376, float %2371, i64 1
  %2378 = insertelement <4 x float> %2377, float %1938, i64 2
  %2379 = insertelement <4 x float> %2378, float %1939, i64 3
  %2380 = select <4 x i1> %1845, <4 x float> %2375, <4 x float> %2379
  %2381 = fmul <4 x float> %2363, %2380
  %2382 = fptrunc <4 x float> %2381 to <4 x half>
  %2383 = extractelement <4 x half> %2382, i64 3
  %2384 = extractelement <4 x half> %2382, i64 2
  %2385 = extractelement <4 x half> %2382, i64 1
  %2386 = extractelement <4 x half> %2382, i64 0
  %2387 = tail call i32 asm sideeffect "", "=v,0"(i32 %29) #9, !range !2, !frisk.register_lifetimes !3
  %2388 = zext nneg i32 %2387 to i64
  %2389 = lshr i64 %2388, 2
  %2390 = and i64 %2389, 48
  %2391 = and i64 %2388, 240
  %2392 = add i64 %26, %2388
  %2393 = sub i64 %2392, %2391
  %2394 = add i64 %2393, %2390
  %2395 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %28
  %2396 = shl nuw nsw i64 %2394, 8
  %2397 = getelementptr inbounds nuw i8, ptr addrspace(1) %2395, i64 %2396
  %2398 = and i64 %2389, 12
  %2399 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %2397, i64 %2398
  store half %2386, ptr addrspace(1) %2399, align 2
  %2400 = or disjoint i64 %2398, 1
  %2401 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %2397, i64 %2400
  store half %2385, ptr addrspace(1) %2401, align 2
  %2402 = or disjoint i64 %2398, 2
  %2403 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %2397, i64 %2402
  store half %2384, ptr addrspace(1) %2403, align 2
  %2404 = or disjoint i64 %2398, 3
  %2405 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %2397, i64 %2404
  store half %2383, ptr addrspace(1) %2405, align 2
  %2406 = or disjoint i64 %2398, 16
  %2407 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %2397, i64 %2406
  store half %2369, ptr addrspace(1) %2407, align 2
  %2408 = or disjoint i64 %2398, 17
  %2409 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %2397, i64 %2408
  store half %2368, ptr addrspace(1) %2409, align 2
  %2410 = or disjoint i64 %2398, 18
  %2411 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %2397, i64 %2410
  store half %2367, ptr addrspace(1) %2411, align 2
  %2412 = or disjoint i64 %2398, 19
  %2413 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %2397, i64 %2412
  store half %2366, ptr addrspace(1) %2413, align 2
  %2414 = getelementptr inbounds nuw i8, ptr addrspace(1) %2397, i64 16384
  %2415 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %2414, i64 %2398
  store half %2351, ptr addrspace(1) %2415, align 2
  %2416 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %2414, i64 %2400
  store half %2350, ptr addrspace(1) %2416, align 2
  %2417 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %2414, i64 %2402
  store half %2349, ptr addrspace(1) %2417, align 2
  %2418 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %2414, i64 %2404
  store half %2348, ptr addrspace(1) %2418, align 2
  %2419 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %2414, i64 %2406
  store half %2334, ptr addrspace(1) %2419, align 2
  %2420 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %2414, i64 %2408
  store half %2333, ptr addrspace(1) %2420, align 2
  %2421 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %2414, i64 %2410
  store half %2332, ptr addrspace(1) %2421, align 2
  %2422 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %2414, i64 %2412
  store half %2331, ptr addrspace(1) %2422, align 2
  %2423 = fmul <4 x float> %2328, %2294
  %2424 = fptrunc <4 x float> %2423 to <4 x half>
  %2425 = extractelement <4 x half> %2424, i64 3
  %2426 = extractelement <4 x half> %2424, i64 2
  %2427 = extractelement <4 x half> %2424, i64 1
  %2428 = extractelement <4 x half> %2424, i64 0
  %2429 = fmul <4 x float> %2328, %2170
  %2430 = fptrunc <4 x float> %2429 to <4 x half>
  %2431 = extractelement <4 x half> %2430, i64 3
  %2432 = extractelement <4 x half> %2430, i64 2
  %2433 = extractelement <4 x half> %2430, i64 1
  %2434 = extractelement <4 x half> %2430, i64 0
  %2435 = fmul <4 x float> %2363, %2046
  %2436 = fptrunc <4 x float> %2435 to <4 x half>
  %2437 = extractelement <4 x half> %2436, i64 3
  %2438 = extractelement <4 x half> %2436, i64 2
  %2439 = extractelement <4 x half> %2436, i64 1
  %2440 = extractelement <4 x half> %2436, i64 0
  %2441 = fmul <4 x float> %2363, %1922
  %2442 = fptrunc <4 x float> %2441 to <4 x half>
  %2443 = extractelement <4 x half> %2442, i64 3
  %2444 = extractelement <4 x half> %2442, i64 2
  %2445 = extractelement <4 x half> %2442, i64 1
  %2446 = extractelement <4 x half> %2442, i64 0
  %2447 = or disjoint i64 %2398, 32
  %2448 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %2397, i64 %2447
  store half %2446, ptr addrspace(1) %2448, align 2
  %2449 = or disjoint i64 %2398, 33
  %2450 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %2397, i64 %2449
  store half %2445, ptr addrspace(1) %2450, align 2
  %2451 = or disjoint i64 %2398, 34
  %2452 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %2397, i64 %2451
  store half %2444, ptr addrspace(1) %2452, align 2
  %2453 = or disjoint i64 %2398, 35
  %2454 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %2397, i64 %2453
  store half %2443, ptr addrspace(1) %2454, align 2
  %2455 = or disjoint i64 %2398, 48
  %2456 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %2397, i64 %2455
  store half %2440, ptr addrspace(1) %2456, align 2
  %2457 = or disjoint i64 %2398, 49
  %2458 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %2397, i64 %2457
  store half %2439, ptr addrspace(1) %2458, align 2
  %2459 = or disjoint i64 %2398, 50
  %2460 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %2397, i64 %2459
  store half %2438, ptr addrspace(1) %2460, align 2
  %2461 = or i64 %2389, 51
  %2462 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %2397, i64 %2461
  store half %2437, ptr addrspace(1) %2462, align 2
  %2463 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %2414, i64 %2447
  store half %2434, ptr addrspace(1) %2463, align 2
  %2464 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %2414, i64 %2449
  store half %2433, ptr addrspace(1) %2464, align 2
  %2465 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %2414, i64 %2451
  store half %2432, ptr addrspace(1) %2465, align 2
  %2466 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %2414, i64 %2453
  store half %2431, ptr addrspace(1) %2466, align 2
  %2467 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %2414, i64 %2455
  store half %2428, ptr addrspace(1) %2467, align 2
  %2468 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %2414, i64 %2457
  store half %2427, ptr addrspace(1) %2468, align 2
  %2469 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %2414, i64 %2459
  store half %2426, ptr addrspace(1) %2469, align 2
  %2470 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %2414, i64 %2461
  store half %2425, ptr addrspace(1) %2470, align 2
  %2471 = fmul <4 x float> %2328, %2261
  %2472 = fptrunc <4 x float> %2471 to <4 x half>
  %2473 = extractelement <4 x half> %2472, i64 3
  %2474 = extractelement <4 x half> %2472, i64 2
  %2475 = extractelement <4 x half> %2472, i64 1
  %2476 = extractelement <4 x half> %2472, i64 0
  %2477 = fmul <4 x float> %2328, %2136
  %2478 = fptrunc <4 x float> %2477 to <4 x half>
  %2479 = extractelement <4 x half> %2478, i64 3
  %2480 = extractelement <4 x half> %2478, i64 2
  %2481 = extractelement <4 x half> %2478, i64 1
  %2482 = extractelement <4 x half> %2478, i64 0
  %2483 = fmul <4 x float> %2363, %2013
  %2484 = fptrunc <4 x float> %2483 to <4 x half>
  %2485 = extractelement <4 x half> %2484, i64 3
  %2486 = extractelement <4 x half> %2484, i64 2
  %2487 = extractelement <4 x half> %2484, i64 1
  %2488 = extractelement <4 x half> %2484, i64 0
  %2489 = fmul <4 x float> %2363, %1888
  %2490 = fptrunc <4 x float> %2489 to <4 x half>
  %2491 = extractelement <4 x half> %2490, i64 3
  %2492 = extractelement <4 x half> %2490, i64 2
  %2493 = extractelement <4 x half> %2490, i64 1
  %2494 = extractelement <4 x half> %2490, i64 0
  %2495 = or disjoint i64 %2398, 64
  %2496 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %2397, i64 %2495
  store half %2494, ptr addrspace(1) %2496, align 2
  %2497 = or disjoint i64 %2398, 65
  %2498 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %2397, i64 %2497
  store half %2493, ptr addrspace(1) %2498, align 2
  %2499 = or disjoint i64 %2398, 66
  %2500 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %2397, i64 %2499
  store half %2492, ptr addrspace(1) %2500, align 2
  %2501 = or disjoint i64 %2398, 67
  %2502 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %2397, i64 %2501
  store half %2491, ptr addrspace(1) %2502, align 2
  %2503 = or disjoint i64 %2398, 80
  %2504 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %2397, i64 %2503
  store half %2488, ptr addrspace(1) %2504, align 2
  %2505 = or disjoint i64 %2398, 81
  %2506 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %2397, i64 %2505
  store half %2487, ptr addrspace(1) %2506, align 2
  %2507 = or disjoint i64 %2398, 82
  %2508 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %2397, i64 %2507
  store half %2486, ptr addrspace(1) %2508, align 2
  %2509 = or disjoint i64 %2398, 83
  %2510 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %2397, i64 %2509
  store half %2485, ptr addrspace(1) %2510, align 2
  %2511 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %2414, i64 %2495
  store half %2482, ptr addrspace(1) %2511, align 2
  %2512 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %2414, i64 %2497
  store half %2481, ptr addrspace(1) %2512, align 2
  %2513 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %2414, i64 %2499
  store half %2480, ptr addrspace(1) %2513, align 2
  %2514 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %2414, i64 %2501
  store half %2479, ptr addrspace(1) %2514, align 2
  %2515 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %2414, i64 %2503
  store half %2476, ptr addrspace(1) %2515, align 2
  %2516 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %2414, i64 %2505
  store half %2475, ptr addrspace(1) %2516, align 2
  %2517 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %2414, i64 %2507
  store half %2474, ptr addrspace(1) %2517, align 2
  %2518 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %2414, i64 %2509
  store half %2473, ptr addrspace(1) %2518, align 2
  %2519 = fmul <4 x float> %2328, %2228
  %2520 = fptrunc <4 x float> %2519 to <4 x half>
  %2521 = extractelement <4 x half> %2520, i64 3
  %2522 = extractelement <4 x half> %2520, i64 2
  %2523 = extractelement <4 x half> %2520, i64 1
  %2524 = extractelement <4 x half> %2520, i64 0
  %2525 = fmul <4 x float> %2328, %2102
  %2526 = fptrunc <4 x float> %2525 to <4 x half>
  %2527 = extractelement <4 x half> %2526, i64 3
  %2528 = extractelement <4 x half> %2526, i64 2
  %2529 = extractelement <4 x half> %2526, i64 1
  %2530 = extractelement <4 x half> %2526, i64 0
  %2531 = fmul <4 x float> %2363, %1980
  %2532 = fptrunc <4 x float> %2531 to <4 x half>
  %2533 = extractelement <4 x half> %2532, i64 3
  %2534 = extractelement <4 x half> %2532, i64 2
  %2535 = extractelement <4 x half> %2532, i64 1
  %2536 = extractelement <4 x half> %2532, i64 0
  %2537 = fmul <4 x float> %2363, %1854
  %2538 = fptrunc <4 x float> %2537 to <4 x half>
  %2539 = extractelement <4 x half> %2538, i64 3
  %2540 = extractelement <4 x half> %2538, i64 2
  %2541 = extractelement <4 x half> %2538, i64 1
  %2542 = extractelement <4 x half> %2538, i64 0
  %2543 = or disjoint i64 %2398, 96
  %2544 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %2397, i64 %2543
  store half %2542, ptr addrspace(1) %2544, align 2
  %2545 = or disjoint i64 %2398, 97
  %2546 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %2397, i64 %2545
  store half %2541, ptr addrspace(1) %2546, align 2
  %2547 = or disjoint i64 %2398, 98
  %2548 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %2397, i64 %2547
  store half %2540, ptr addrspace(1) %2548, align 2
  %2549 = or disjoint i64 %2398, 99
  %2550 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %2397, i64 %2549
  store half %2539, ptr addrspace(1) %2550, align 2
  %2551 = or disjoint i64 %2398, 112
  %2552 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %2397, i64 %2551
  store half %2536, ptr addrspace(1) %2552, align 2
  %2553 = or disjoint i64 %2398, 113
  %2554 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %2397, i64 %2553
  store half %2535, ptr addrspace(1) %2554, align 2
  %2555 = or disjoint i64 %2398, 114
  %2556 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %2397, i64 %2555
  store half %2534, ptr addrspace(1) %2556, align 2
  %2557 = or i64 %2389, 115
  %2558 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %2397, i64 %2557
  store half %2533, ptr addrspace(1) %2558, align 2
  %2559 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %2414, i64 %2543
  store half %2530, ptr addrspace(1) %2559, align 2
  %2560 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %2414, i64 %2545
  store half %2529, ptr addrspace(1) %2560, align 2
  %2561 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %2414, i64 %2547
  store half %2528, ptr addrspace(1) %2561, align 2
  %2562 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %2414, i64 %2549
  store half %2527, ptr addrspace(1) %2562, align 2
  %2563 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %2414, i64 %2551
  store half %2524, ptr addrspace(1) %2563, align 2
  %2564 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %2414, i64 %2553
  store half %2523, ptr addrspace(1) %2564, align 2
  %2565 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %2414, i64 %2555
  store half %2522, ptr addrspace(1) %2565, align 2
  %2566 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %2414, i64 %2557
  store half %2521, ptr addrspace(1) %2566, align 2
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.x() #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.y() #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #2

; Function Attrs: convergent mustprogress nocallback nocreateundeforpoison nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.readfirstlane.i32(i32) #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.z() #2

; Function Attrs: convergent mustprogress nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.barrier() #4

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.lo(i32, i32) #5

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.hi(i32, i32) #5

; Function Attrs: convergent mustprogress nocallback nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.ds.bpermute(i32, i32) #6

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.maxnum.f32(float, float) #7

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare <1 x float> @llvm.maxnum.v1f32(<1 x float>, <1 x float>) #7

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.amdgcn.exp2.f32(float) #7

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.vector.reduce.fadd.v4f32(float, <4 x float>) #8

attributes #0 = { convergent mustprogress nofree willreturn memory(read) }
attributes #1 = { "amdgpu-flat-work-group-size"="256,256" "amdgpu-waves-per-eu"="1" "frisk.global_prefetch"="8" "frisk.register_comparisons"="1" "frisk.register_lifetimes"="1" "uniform-work-group-size" }
attributes #2 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { convergent mustprogress nocallback nocreateundeforpoison nofree nounwind willreturn memory(none) }
attributes #4 = { convergent mustprogress nocallback nofree nounwind willreturn }
attributes #5 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none) }
attributes #6 = { convergent mustprogress nocallback nofree nounwind willreturn memory(none) }
attributes #7 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #8 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #9 = { convergent nounwind willreturn memory(none) }
attributes #10 = { convergent nounwind }
attributes #11 = { convergent willreturn memory(read) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 256, i32 1, i32 1}
!2 = !{i32 0, i32 256}
!3 = !{}
