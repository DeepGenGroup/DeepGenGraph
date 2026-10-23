target triple = "amdgcn-amd-amdhsa"
@shm_0 = addrspace(3) global [16384 x i8] undef, align 16
declare i32 @llvm.amdgcn.workitem.id.x() #0
define amdgpu_kernel void @probe(ptr addrspace(1) %out) #1 {
  %probe.ssa24 = call i32 @llvm.amdgcn.workitem.id.x()
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %probe.ssa24) #2
  %address.local179 = lshr i32 %address.thread, 2
  %address.local180 = and i32 %address.local179, 4
  %address.local181 = shl nuw nsw i32 %address.thread, 1
  %address.local182 = and i32 %address.local181, 24
  %address.local183 = or disjoint i32 %address.local180, %address.local182
  %address.local184 = shl nuw nsw i32 %address.thread, 5
  %address.local185 = and i32 %address.local184, 1024
  %address.local186 = or disjoint i32 %address.local183, %address.local185
  %address.local187 = shl nuw nsw i32 %address.thread, 7
  %address.local188 = and i32 %address.local187, 384
  %address.local189 = or disjoint i32 %address.local186, %address.local188
  %address.local190 = zext nneg i32 %address.local189 to i64
  %address.local191 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local190
  %ksw.read0.address = ptrtoint ptr addrspace(3) %address.local191 to i32
  %ksw.read.bit = and i32 %probe.ssa24, 1
  %ksw.read.mask = shl i32 %ksw.read.bit, 6
  %ksw.read0.xor = xor i32 %ksw.read0.address, %ksw.read.mask
  %ksw.read0.pointer = inttoptr i32 %ksw.read0.xor to ptr addrspace(3)
  %address.local192 = lshr i32 %address.thread, 2
  %address.local193 = and i32 %address.local192, 4
  %address.local194 = shl nuw nsw i32 %address.thread, 1
  %address.local195 = and i32 %address.local194, 24
  %address.local196 = or disjoint i32 %address.local193, %address.local195
  %address.local197 = shl nuw nsw i32 %address.thread, 5
  %address.local198 = and i32 %address.local197, 1024
  %address.local199 = or disjoint i32 %address.local196, %address.local198
  %address.local200 = shl nuw nsw i32 %address.thread, 7
  %address.local201 = and i32 %address.local200, 384
  %address.local202 = or disjoint i32 %address.local199, %address.local201
  %address.local203 = zext nneg i32 %address.local202 to i64
  %address.local204 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local203
  %address.local205 = getelementptr i8, ptr addrspace(3) %address.local204, i64 4096
  %ksw.read1.address = ptrtoint ptr addrspace(3) %address.local205 to i32
  %ksw.read1.xor = xor i32 %ksw.read1.address, %ksw.read.mask
  %ksw.read1.pointer = inttoptr i32 %ksw.read1.xor to ptr addrspace(3)
  %address.local = zext nneg i32 %address.thread to i64
  %address.local206 = shl nuw nsw i64 %address.local, 7
  %address.local207 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local206
  %address.local208 = lshr i64 %address.local, 2
  %address.local209 = mul nsw i64 %address.local208, -512
  %address.local210 = getelementptr [2 x i8], ptr addrspace(3) %address.local207, i64 %address.local209
  %address.local211 = shl nuw nsw i64 %address.local, 1
  %address.local212 = and i64 %address.local211, 24
  %address.local213 = getelementptr [2 x i8], ptr addrspace(3) %address.local210, i64 %address.local212
  %address.local214 = lshr i64 %address.local, 2
  %address.local215 = and i64 %address.local214, 12
  %address.local216 = getelementptr [2 x i8], ptr addrspace(3) %address.local213, i64 %address.local215
  %address.local217 = and i64 %address.local, 63
  %address.local218 = lshr i64 %address.local217, 5
  %address.local219 = mul nuw nsw i64 %address.local218, 1016
  %address.local220 = getelementptr [2 x i8], ptr addrspace(3) %address.local216, i64 %address.local219
  %address.local221 = getelementptr i8, ptr addrspace(3) %address.local220, i64 64
  %ksw.read2.address = ptrtoint ptr addrspace(3) %address.local221 to i32
  %ksw.read2.xor = xor i32 %ksw.read2.address, %ksw.read.mask
  %ksw.read2.pointer = inttoptr i32 %ksw.read2.xor to ptr addrspace(3)
  %address.local222 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local206
  %address.local223 = getelementptr [2 x i8], ptr addrspace(3) %address.local222, i64 %address.local215
  %address.local224 = getelementptr [2 x i8], ptr addrspace(3) %address.local223, i64 %address.local219
  %address.local225 = getelementptr [2 x i8], ptr addrspace(3) %address.local224, i64 %address.local209
  %address.local226 = getelementptr [2 x i8], ptr addrspace(3) %address.local225, i64 %address.local212
  %address.local227 = getelementptr i8, ptr addrspace(3) %address.local226, i64 4160
  %ksw.read3.address = ptrtoint ptr addrspace(3) %address.local227 to i32
  %ksw.read3.xor = xor i32 %ksw.read3.address, %ksw.read.mask
  %ksw.read3.pointer = inttoptr i32 %ksw.read3.xor to ptr addrspace(3)
  %address.local228 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local206
  %address.local229 = getelementptr [2 x i8], ptr addrspace(3) %address.local228, i64 %address.local209
  %address.local230 = getelementptr [2 x i8], ptr addrspace(3) %address.local229, i64 %address.local212
  %address.local231 = getelementptr [2 x i8], ptr addrspace(3) %address.local230, i64 %address.local215
  %address.local232 = getelementptr [2 x i8], ptr addrspace(3) %address.local231, i64 %address.local219
  %address.local233 = getelementptr i8, ptr addrspace(3) %address.local232, i64 128
  %ksw.read4.address = ptrtoint ptr addrspace(3) %address.local233 to i32
  %ksw.read4.xor = xor i32 %ksw.read4.address, %ksw.read.mask
  %ksw.read4.pointer = inttoptr i32 %ksw.read4.xor to ptr addrspace(3)
  %address.local234 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local206
  %address.local235 = getelementptr [2 x i8], ptr addrspace(3) %address.local234, i64 %address.local215
  %address.local236 = getelementptr [2 x i8], ptr addrspace(3) %address.local235, i64 %address.local219
  %address.local237 = getelementptr [2 x i8], ptr addrspace(3) %address.local236, i64 %address.local209
  %address.local238 = getelementptr [2 x i8], ptr addrspace(3) %address.local237, i64 %address.local212
  %address.local239 = getelementptr i8, ptr addrspace(3) %address.local238, i64 4224
  %ksw.read5.address = ptrtoint ptr addrspace(3) %address.local239 to i32
  %ksw.read5.xor = xor i32 %ksw.read5.address, %ksw.read.mask
  %ksw.read5.pointer = inttoptr i32 %ksw.read5.xor to ptr addrspace(3)
  %address.local240 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local206
  %address.local241 = getelementptr [2 x i8], ptr addrspace(3) %address.local240, i64 %address.local209
  %address.local242 = getelementptr [2 x i8], ptr addrspace(3) %address.local241, i64 %address.local212
  %address.local243 = getelementptr [2 x i8], ptr addrspace(3) %address.local242, i64 %address.local215
  %address.local244 = getelementptr [2 x i8], ptr addrspace(3) %address.local243, i64 %address.local219
  %address.local245 = getelementptr i8, ptr addrspace(3) %address.local244, i64 192
  %ksw.read6.address = ptrtoint ptr addrspace(3) %address.local245 to i32
  %ksw.read6.xor = xor i32 %ksw.read6.address, %ksw.read.mask
  %ksw.read6.pointer = inttoptr i32 %ksw.read6.xor to ptr addrspace(3)
  %address.local246 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local206
  %address.local247 = getelementptr [2 x i8], ptr addrspace(3) %address.local246, i64 %address.local215
  %address.local248 = getelementptr [2 x i8], ptr addrspace(3) %address.local247, i64 %address.local219
  %address.local249 = getelementptr [2 x i8], ptr addrspace(3) %address.local248, i64 %address.local209
  %address.local250 = getelementptr [2 x i8], ptr addrspace(3) %address.local249, i64 %address.local212
  %address.local251 = getelementptr i8, ptr addrspace(3) %address.local250, i64 4288
  %ksw.read7.address = ptrtoint ptr addrspace(3) %address.local251 to i32
  %ksw.read7.xor = xor i32 %ksw.read7.address, %ksw.read.mask
  %ksw.read7.pointer = inttoptr i32 %ksw.read7.xor to ptr addrspace(3)
  %address.local252 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local190
  %address.local253 = getelementptr i8, ptr addrspace(3) %address.local252, i64 1024
  %ksw.read8.address = ptrtoint ptr addrspace(3) %address.local253 to i32
  %ksw.read8.xor = xor i32 %ksw.read8.address, %ksw.read.mask
  %ksw.read8.pointer = inttoptr i32 %ksw.read8.xor to ptr addrspace(3)
  %address.local254 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local203
  %address.local255 = getelementptr i8, ptr addrspace(3) %address.local254, i64 5120
  %ksw.read9.address = ptrtoint ptr addrspace(3) %address.local255 to i32
  %ksw.read9.xor = xor i32 %ksw.read9.address, %ksw.read.mask
  %ksw.read9.pointer = inttoptr i32 %ksw.read9.xor to ptr addrspace(3)
  %address.local256 = add nuw nsw i64 %address.local206, 512
  %address.local257 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local256
  %address.local258 = getelementptr [2 x i8], ptr addrspace(3) %address.local257, i64 %address.local209
  %address.local259 = getelementptr [2 x i8], ptr addrspace(3) %address.local258, i64 %address.local212
  %address.local260 = getelementptr [2 x i8], ptr addrspace(3) %address.local259, i64 %address.local215
  %address.local261 = getelementptr [2 x i8], ptr addrspace(3) %address.local260, i64 %address.local219
  %address.local262 = getelementptr i8, ptr addrspace(3) %address.local261, i64 64
  %ksw.read10.address = ptrtoint ptr addrspace(3) %address.local262 to i32
  %ksw.read10.xor = xor i32 %ksw.read10.address, %ksw.read.mask
  %ksw.read10.pointer = inttoptr i32 %ksw.read10.xor to ptr addrspace(3)
  %address.local263 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local256
  %address.local264 = getelementptr [2 x i8], ptr addrspace(3) %address.local263, i64 %address.local215
  %address.local265 = getelementptr [2 x i8], ptr addrspace(3) %address.local264, i64 %address.local219
  %address.local266 = getelementptr [2 x i8], ptr addrspace(3) %address.local265, i64 %address.local209
  %address.local267 = getelementptr [2 x i8], ptr addrspace(3) %address.local266, i64 %address.local212
  %address.local268 = getelementptr i8, ptr addrspace(3) %address.local267, i64 4160
  %ksw.read11.address = ptrtoint ptr addrspace(3) %address.local268 to i32
  %ksw.read11.xor = xor i32 %ksw.read11.address, %ksw.read.mask
  %ksw.read11.pointer = inttoptr i32 %ksw.read11.xor to ptr addrspace(3)
  %address.local269 = add nuw nsw i64 %address.local206, 512
  %address.local270 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local269
  %address.local271 = getelementptr [2 x i8], ptr addrspace(3) %address.local270, i64 %address.local209
  %address.local272 = getelementptr [2 x i8], ptr addrspace(3) %address.local271, i64 %address.local212
  %address.local273 = getelementptr [2 x i8], ptr addrspace(3) %address.local272, i64 %address.local215
  %address.local274 = getelementptr [2 x i8], ptr addrspace(3) %address.local273, i64 %address.local219
  %address.local275 = getelementptr i8, ptr addrspace(3) %address.local274, i64 128
  %ksw.read12.address = ptrtoint ptr addrspace(3) %address.local275 to i32
  %ksw.read12.xor = xor i32 %ksw.read12.address, %ksw.read.mask
  %ksw.read12.pointer = inttoptr i32 %ksw.read12.xor to ptr addrspace(3)
  %address.local276 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local269
  %address.local277 = getelementptr [2 x i8], ptr addrspace(3) %address.local276, i64 %address.local215
  %address.local278 = getelementptr [2 x i8], ptr addrspace(3) %address.local277, i64 %address.local219
  %address.local279 = getelementptr [2 x i8], ptr addrspace(3) %address.local278, i64 %address.local209
  %address.local280 = getelementptr [2 x i8], ptr addrspace(3) %address.local279, i64 %address.local212
  %address.local281 = getelementptr i8, ptr addrspace(3) %address.local280, i64 4224
  %ksw.read13.address = ptrtoint ptr addrspace(3) %address.local281 to i32
  %ksw.read13.xor = xor i32 %ksw.read13.address, %ksw.read.mask
  %ksw.read13.pointer = inttoptr i32 %ksw.read13.xor to ptr addrspace(3)
  %address.local282 = add nuw nsw i64 %address.local206, 512
  %address.local283 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local282
  %address.local284 = getelementptr [2 x i8], ptr addrspace(3) %address.local283, i64 %address.local209
  %address.local285 = getelementptr [2 x i8], ptr addrspace(3) %address.local284, i64 %address.local212
  %address.local286 = getelementptr [2 x i8], ptr addrspace(3) %address.local285, i64 %address.local215
  %address.local287 = getelementptr [2 x i8], ptr addrspace(3) %address.local286, i64 %address.local219
  %address.local288 = getelementptr i8, ptr addrspace(3) %address.local287, i64 192
  %ksw.read14.address = ptrtoint ptr addrspace(3) %address.local288 to i32
  %ksw.read14.xor = xor i32 %ksw.read14.address, %ksw.read.mask
  %ksw.read14.pointer = inttoptr i32 %ksw.read14.xor to ptr addrspace(3)
  %address.local289 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local282
  %address.local290 = getelementptr [2 x i8], ptr addrspace(3) %address.local289, i64 %address.local215
  %address.local291 = getelementptr [2 x i8], ptr addrspace(3) %address.local290, i64 %address.local219
  %address.local292 = getelementptr [2 x i8], ptr addrspace(3) %address.local291, i64 %address.local209
  %address.local293 = getelementptr [2 x i8], ptr addrspace(3) %address.local292, i64 %address.local212
  %address.local294 = getelementptr i8, ptr addrspace(3) %address.local293, i64 4288
  %ksw.read15.address = ptrtoint ptr addrspace(3) %address.local294 to i32
  %ksw.read15.xor = xor i32 %ksw.read15.address, %ksw.read.mask
  %ksw.read15.pointer = inttoptr i32 %ksw.read15.xor to ptr addrspace(3)
  %probe.value0 = load <4 x half>, ptr addrspace(3) %ksw.read0.pointer, align 2
  %probe.value1 = load <4 x half>, ptr addrspace(3) %ksw.read1.pointer, align 2
  %probe.value2 = load <4 x half>, ptr addrspace(3) %ksw.read2.pointer, align 2
  %probe.value3 = load <4 x half>, ptr addrspace(3) %ksw.read3.pointer, align 2
  %probe.value4 = load <4 x half>, ptr addrspace(3) %ksw.read4.pointer, align 2
  %probe.value5 = load <4 x half>, ptr addrspace(3) %ksw.read5.pointer, align 2
  %probe.value6 = load <4 x half>, ptr addrspace(3) %ksw.read6.pointer, align 2
  %probe.value7 = load <4 x half>, ptr addrspace(3) %ksw.read7.pointer, align 2
  %probe.value8 = load <4 x half>, ptr addrspace(3) %ksw.read8.pointer, align 2
  %probe.value9 = load <4 x half>, ptr addrspace(3) %ksw.read9.pointer, align 2
  %probe.value10 = load <4 x half>, ptr addrspace(3) %ksw.read10.pointer, align 2
  %probe.value11 = load <4 x half>, ptr addrspace(3) %ksw.read11.pointer, align 2
  %probe.value12 = load <4 x half>, ptr addrspace(3) %ksw.read12.pointer, align 2
  %probe.value13 = load <4 x half>, ptr addrspace(3) %ksw.read13.pointer, align 2
  %probe.value14 = load <4 x half>, ptr addrspace(3) %ksw.read14.pointer, align 2
  %probe.value15 = load <4 x half>, ptr addrspace(3) %ksw.read15.pointer, align 2
  %probe.output.offset = shl i32 %probe.ssa24, 7
  %probe.output.base = getelementptr i8, ptr addrspace(1) %out, i32 %probe.output.offset
  %probe.out0 = getelementptr i8, ptr addrspace(1) %probe.output.base, i32 0
  store <4 x half> %probe.value0, ptr addrspace(1) %probe.out0, align 8
  %probe.out1 = getelementptr i8, ptr addrspace(1) %probe.output.base, i32 8
  store <4 x half> %probe.value1, ptr addrspace(1) %probe.out1, align 8
  %probe.out2 = getelementptr i8, ptr addrspace(1) %probe.output.base, i32 16
  store <4 x half> %probe.value2, ptr addrspace(1) %probe.out2, align 8
  %probe.out3 = getelementptr i8, ptr addrspace(1) %probe.output.base, i32 24
  store <4 x half> %probe.value3, ptr addrspace(1) %probe.out3, align 8
  %probe.out4 = getelementptr i8, ptr addrspace(1) %probe.output.base, i32 32
  store <4 x half> %probe.value4, ptr addrspace(1) %probe.out4, align 8
  %probe.out5 = getelementptr i8, ptr addrspace(1) %probe.output.base, i32 40
  store <4 x half> %probe.value5, ptr addrspace(1) %probe.out5, align 8
  %probe.out6 = getelementptr i8, ptr addrspace(1) %probe.output.base, i32 48
  store <4 x half> %probe.value6, ptr addrspace(1) %probe.out6, align 8
  %probe.out7 = getelementptr i8, ptr addrspace(1) %probe.output.base, i32 56
  store <4 x half> %probe.value7, ptr addrspace(1) %probe.out7, align 8
  %probe.out8 = getelementptr i8, ptr addrspace(1) %probe.output.base, i32 64
  store <4 x half> %probe.value8, ptr addrspace(1) %probe.out8, align 8
  %probe.out9 = getelementptr i8, ptr addrspace(1) %probe.output.base, i32 72
  store <4 x half> %probe.value9, ptr addrspace(1) %probe.out9, align 8
  %probe.out10 = getelementptr i8, ptr addrspace(1) %probe.output.base, i32 80
  store <4 x half> %probe.value10, ptr addrspace(1) %probe.out10, align 8
  %probe.out11 = getelementptr i8, ptr addrspace(1) %probe.output.base, i32 88
  store <4 x half> %probe.value11, ptr addrspace(1) %probe.out11, align 8
  %probe.out12 = getelementptr i8, ptr addrspace(1) %probe.output.base, i32 96
  store <4 x half> %probe.value12, ptr addrspace(1) %probe.out12, align 8
  %probe.out13 = getelementptr i8, ptr addrspace(1) %probe.output.base, i32 104
  store <4 x half> %probe.value13, ptr addrspace(1) %probe.out13, align 8
  %probe.out14 = getelementptr i8, ptr addrspace(1) %probe.output.base, i32 112
  store <4 x half> %probe.value14, ptr addrspace(1) %probe.out14, align 8
  %probe.out15 = getelementptr i8, ptr addrspace(1) %probe.output.base, i32 120
  store <4 x half> %probe.value15, ptr addrspace(1) %probe.out15, align 8
  ret void
}
attributes #0 = { nounwind memory(none) }
attributes #1 = { "amdgpu-flat-work-group-size"="256,256" }
attributes #2 = { convergent nounwind memory(none) }
