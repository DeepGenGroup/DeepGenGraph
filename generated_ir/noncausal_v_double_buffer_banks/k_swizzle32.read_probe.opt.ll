; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/noncausal_v_double_buffer_banks/k_swizzle32.read_probe.ll'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/noncausal_v_double_buffer_banks/k_swizzle32.read_probe.ll"
target datalayout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-p10:32:32-p11:32:32-p12:32:32-p13:32:32-p14:32:32-p15:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

@shm_0 = addrspace(3) global [16384 x i8] undef, align 16

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #0

; Function Attrs: nounwind memory(read, argmem: readwrite, inaccessiblemem: none, target_mem: none)
define amdgpu_kernel void @probe(ptr addrspace(1) nofree writeonly captures(none) %probe.out) local_unnamed_addr #1 {
  %grp.n25 = tail call i32 @llvm.amdgcn.workitem.id.x()
  %address.thread498 = tail call i32 asm sideeffect "", "=v,0"(i32 %grp.n25) #2
  %address.local499 = zext nneg i32 %address.thread498 to i64
  %.idx = mul i32 %address.thread498, 320
  %grp.n154 = getelementptr i8, ptr addrspace(3) @shm_0, i32 %.idx
  %address.local508 = lshr i32 %address.thread498, 2
  %1 = mul i32 %address.local508, -1280
  %grp.n155 = getelementptr i8, ptr addrspace(3) %grp.n154, i32 %1
  %address.local545 = shl i32 %address.thread498, 1
  %address.local546 = and i32 %address.local545, 24
  %grp.n156 = getelementptr [2 x i8], ptr addrspace(3) %grp.n155, i32 %address.local546
  %address.local548 = shl i64 %address.local499, 62
  %address.local549 = ashr exact i64 %address.local548, 57
  %2 = trunc nsw i64 %address.local549 to i32
  %3 = and i32 %2, -64
  %grp.n157 = getelementptr [2 x i8], ptr addrspace(3) %grp.n156, i32 %3
  %4 = and i32 %address.local508, 12
  %grp.n158 = getelementptr [2 x i8], ptr addrspace(3) %grp.n157, i32 %4
  %5 = and i64 %address.local499, 32
  %.not = icmp eq i64 %5, 0
  %6 = select i1 %.not, i32 0, i32 1016
  %address.local559 = getelementptr [2 x i8], ptr addrspace(3) %grp.n158, i32 %6
  %bank.read0.addr = ptrtoint ptr addrspace(3) %address.local559 to i32
  %bank.read0.shift = lshr i32 %bank.read0.addr, 6
  %bank.read0.mask = and i32 %bank.read0.shift, 8
  %bank.read0.xor = xor i32 %bank.read0.mask, %bank.read0.addr
  %bank.read0.ptr = inttoptr i32 %bank.read0.xor to ptr addrspace(3)
  %lds.const.gep1210 = getelementptr i8, ptr addrspace(3) %address.local559, i32 4096
  %bank.read1.addr = ptrtoint ptr addrspace(3) %lds.const.gep1210 to i32
  %bank.read1.shift = lshr i32 %bank.read1.addr, 6
  %bank.read1.mask = and i32 %bank.read1.shift, 8
  %bank.read1.xor = xor i32 %bank.read1.mask, %bank.read1.addr
  %bank.read1.ptr = inttoptr i32 %bank.read1.xor to ptr addrspace(3)
  %address.local572 = shl nuw nsw i32 %address.thread498, 8
  %address.local573 = and i32 %address.local572, 768
  %address.local575 = shl nuw nsw i32 %address.thread498, 5
  %address.local576 = and i32 %address.local575, 96
  %grp.n115 = and i32 %grp.n25, 1
  %grp.n1161 = add nuw nsw i32 %grp.n115, 1
  %probe.ssa16 = shl nuw nsw i32 %grp.n25, 2
  %probe.ssa17 = and i32 %probe.ssa16, 12
  %narrow = or disjoint i32 %grp.n1161, %probe.ssa17
  %probe.ssa18 = lshr i32 %narrow, 1
  %grp.n118 = zext nneg i32 %probe.ssa18 to i64
  %address.base = tail call i64 asm sideeffect "", "=v,0"(i64 %grp.n118) #2
  %7 = shl i32 %address.thread498, 30
  %8 = ashr i32 %7, 31
  %9 = trunc i64 %address.base to i32
  %address.local582.tr.neg = sub i32 %8, %9
  %10 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i32 %address.local573
  %11 = getelementptr [2 x i8], ptr addrspace(3) %10, i32 %address.local576
  %12 = getelementptr [2 x i8], ptr addrspace(3) %11, i32 %address.local546
  %grp.n162.idx = shl i32 %address.local582.tr.neg, 7
  %grp.n162 = getelementptr i8, ptr addrspace(3) %12, i32 %grp.n162.idx
  %grp.n163 = getelementptr [2 x i8], ptr addrspace(3) %grp.n162, i32 %4
  %grp.n164 = getelementptr [2 x i8], ptr addrspace(3) %grp.n163, i32 %6
  %address.local588 = getelementptr i8, ptr addrspace(3) %grp.n164, i32 64
  %bank.read2.addr = ptrtoint ptr addrspace(3) %address.local588 to i32
  %bank.read2.shift = lshr i32 %bank.read2.addr, 6
  %bank.read2.mask = and i32 %bank.read2.shift, 8
  %bank.read2.xor = xor i32 %bank.read2.mask, %bank.read2.addr
  %bank.read2.ptr = inttoptr i32 %bank.read2.xor to ptr addrspace(3)
  %lds.const.gep1214 = getelementptr i8, ptr addrspace(3) %grp.n164, i32 192
  %bank.read3.addr = ptrtoint ptr addrspace(3) %lds.const.gep1214 to i32
  %bank.read3.shift = lshr i32 %bank.read3.addr, 6
  %bank.read3.mask = and i32 %bank.read3.shift, 8
  %bank.read3.xor = xor i32 %bank.read3.mask, %bank.read3.addr
  %bank.read3.ptr = inttoptr i32 %bank.read3.xor to ptr addrspace(3)
  %lds.const.gep1211 = getelementptr i8, ptr addrspace(3) %grp.n164, i32 4160
  %bank.read4.addr = ptrtoint ptr addrspace(3) %lds.const.gep1211 to i32
  %bank.read4.shift = lshr i32 %bank.read4.addr, 6
  %bank.read4.mask = and i32 %bank.read4.shift, 8
  %bank.read4.xor = xor i32 %bank.read4.mask, %bank.read4.addr
  %bank.read4.ptr = inttoptr i32 %bank.read4.xor to ptr addrspace(3)
  %lds.const.gep1215 = getelementptr i8, ptr addrspace(3) %grp.n164, i32 4288
  %bank.read5.addr = ptrtoint ptr addrspace(3) %lds.const.gep1215 to i32
  %bank.read5.shift = lshr i32 %bank.read5.addr, 6
  %bank.read5.mask = and i32 %bank.read5.shift, 8
  %bank.read5.xor = xor i32 %bank.read5.mask, %bank.read5.addr
  %bank.read5.ptr = inttoptr i32 %bank.read5.xor to ptr addrspace(3)
  %lds.const.gep1212 = getelementptr i8, ptr addrspace(3) %address.local559, i32 128
  %bank.read6.addr = ptrtoint ptr addrspace(3) %lds.const.gep1212 to i32
  %bank.read6.shift = lshr i32 %bank.read6.addr, 6
  %bank.read6.mask = and i32 %bank.read6.shift, 8
  %bank.read6.xor = xor i32 %bank.read6.mask, %bank.read6.addr
  %bank.read6.ptr = inttoptr i32 %bank.read6.xor to ptr addrspace(3)
  %lds.const.gep1216 = getelementptr i8, ptr addrspace(3) %address.local559, i32 1024
  %bank.read7.addr = ptrtoint ptr addrspace(3) %lds.const.gep1216 to i32
  %bank.read7.shift = lshr i32 %bank.read7.addr, 6
  %bank.read7.mask = and i32 %bank.read7.shift, 8
  %bank.read7.xor = xor i32 %bank.read7.mask, %bank.read7.addr
  %bank.read7.ptr = inttoptr i32 %bank.read7.xor to ptr addrspace(3)
  %lds.const.gep1213 = getelementptr i8, ptr addrspace(3) %address.local559, i32 4224
  %bank.read8.addr = ptrtoint ptr addrspace(3) %lds.const.gep1213 to i32
  %bank.read8.shift = lshr i32 %bank.read8.addr, 6
  %bank.read8.mask = and i32 %bank.read8.shift, 8
  %bank.read8.xor = xor i32 %bank.read8.mask, %bank.read8.addr
  %bank.read8.ptr = inttoptr i32 %bank.read8.xor to ptr addrspace(3)
  %lds.const.gep1220 = getelementptr i8, ptr addrspace(3) %address.local559, i32 1152
  %bank.read9.addr = ptrtoint ptr addrspace(3) %lds.const.gep1220 to i32
  %bank.read9.shift = lshr i32 %bank.read9.addr, 6
  %bank.read9.mask = and i32 %bank.read9.shift, 8
  %bank.read9.xor = xor i32 %bank.read9.mask, %bank.read9.addr
  %bank.read9.ptr = inttoptr i32 %bank.read9.xor to ptr addrspace(3)
  %lds.const.gep1217 = getelementptr i8, ptr addrspace(3) %address.local559, i32 5120
  %bank.read10.addr = ptrtoint ptr addrspace(3) %lds.const.gep1217 to i32
  %bank.read10.shift = lshr i32 %bank.read10.addr, 6
  %bank.read10.mask = and i32 %bank.read10.shift, 8
  %bank.read10.xor = xor i32 %bank.read10.mask, %bank.read10.addr
  %bank.read10.ptr = inttoptr i32 %bank.read10.xor to ptr addrspace(3)
  %lds.const.gep1221 = getelementptr i8, ptr addrspace(3) %address.local559, i32 5248
  %bank.read11.addr = ptrtoint ptr addrspace(3) %lds.const.gep1221 to i32
  %bank.read11.shift = lshr i32 %bank.read11.addr, 6
  %bank.read11.mask = and i32 %bank.read11.shift, 8
  %bank.read11.xor = xor i32 %bank.read11.mask, %bank.read11.addr
  %bank.read11.ptr = inttoptr i32 %bank.read11.xor to ptr addrspace(3)
  %lds.const.gep1218 = getelementptr i8, ptr addrspace(3) %grp.n164, i32 1088
  %bank.read12.addr = ptrtoint ptr addrspace(3) %lds.const.gep1218 to i32
  %bank.read12.shift = lshr i32 %bank.read12.addr, 6
  %bank.read12.mask = and i32 %bank.read12.shift, 8
  %bank.read12.xor = xor i32 %bank.read12.mask, %bank.read12.addr
  %bank.read12.ptr = inttoptr i32 %bank.read12.xor to ptr addrspace(3)
  %lds.const.gep1222 = getelementptr i8, ptr addrspace(3) %grp.n164, i32 1216
  %bank.read13.addr = ptrtoint ptr addrspace(3) %lds.const.gep1222 to i32
  %bank.read13.shift = lshr i32 %bank.read13.addr, 6
  %bank.read13.mask = and i32 %bank.read13.shift, 8
  %bank.read13.xor = xor i32 %bank.read13.mask, %bank.read13.addr
  %bank.read13.ptr = inttoptr i32 %bank.read13.xor to ptr addrspace(3)
  %lds.const.gep1219 = getelementptr i8, ptr addrspace(3) %grp.n164, i32 5184
  %bank.read14.addr = ptrtoint ptr addrspace(3) %lds.const.gep1219 to i32
  %bank.read14.shift = lshr i32 %bank.read14.addr, 6
  %bank.read14.mask = and i32 %bank.read14.shift, 8
  %bank.read14.xor = xor i32 %bank.read14.mask, %bank.read14.addr
  %bank.read14.ptr = inttoptr i32 %bank.read14.xor to ptr addrspace(3)
  %lds.const.gep1223 = getelementptr i8, ptr addrspace(3) %grp.n164, i32 5312
  %bank.read15.addr = ptrtoint ptr addrspace(3) %lds.const.gep1223 to i32
  %bank.read15.shift = lshr i32 %bank.read15.addr, 6
  %bank.read15.mask = and i32 %bank.read15.shift, 8
  %bank.read15.xor = xor i32 %bank.read15.mask, %bank.read15.addr
  %bank.read15.ptr = inttoptr i32 %bank.read15.xor to ptr addrspace(3)
  %lds.const.gep1238 = getelementptr i8, ptr addrspace(3) %address.local559, i32 8192
  %bank.read16.addr = ptrtoint ptr addrspace(3) %lds.const.gep1238 to i32
  %bank.read16.shift = lshr i32 %bank.read16.addr, 6
  %bank.read16.mask = and i32 %bank.read16.shift, 8
  %bank.read16.xor = xor i32 %bank.read16.mask, %bank.read16.addr
  %bank.read16.ptr = inttoptr i32 %bank.read16.xor to ptr addrspace(3)
  %lds.const.gep1242 = getelementptr i8, ptr addrspace(3) %address.local559, i32 8320
  %bank.read17.addr = ptrtoint ptr addrspace(3) %lds.const.gep1242 to i32
  %bank.read17.shift = lshr i32 %bank.read17.addr, 6
  %bank.read17.mask = and i32 %bank.read17.shift, 8
  %bank.read17.xor = xor i32 %bank.read17.mask, %bank.read17.addr
  %bank.read17.ptr = inttoptr i32 %bank.read17.xor to ptr addrspace(3)
  %lds.const.gep1239 = getelementptr i8, ptr addrspace(3) %address.local559, i32 12288
  %bank.read18.addr = ptrtoint ptr addrspace(3) %lds.const.gep1239 to i32
  %bank.read18.shift = lshr i32 %bank.read18.addr, 6
  %bank.read18.mask = and i32 %bank.read18.shift, 8
  %bank.read18.xor = xor i32 %bank.read18.mask, %bank.read18.addr
  %bank.read18.ptr = inttoptr i32 %bank.read18.xor to ptr addrspace(3)
  %lds.const.gep1243 = getelementptr i8, ptr addrspace(3) %address.local559, i32 12416
  %bank.read19.addr = ptrtoint ptr addrspace(3) %lds.const.gep1243 to i32
  %bank.read19.shift = lshr i32 %bank.read19.addr, 6
  %bank.read19.mask = and i32 %bank.read19.shift, 8
  %bank.read19.xor = xor i32 %bank.read19.mask, %bank.read19.addr
  %bank.read19.ptr = inttoptr i32 %bank.read19.xor to ptr addrspace(3)
  %lds.const.gep1240 = getelementptr i8, ptr addrspace(3) %grp.n164, i32 8256
  %bank.read20.addr = ptrtoint ptr addrspace(3) %lds.const.gep1240 to i32
  %bank.read20.shift = lshr i32 %bank.read20.addr, 6
  %bank.read20.mask = and i32 %bank.read20.shift, 8
  %bank.read20.xor = xor i32 %bank.read20.mask, %bank.read20.addr
  %bank.read20.ptr = inttoptr i32 %bank.read20.xor to ptr addrspace(3)
  %lds.const.gep1244 = getelementptr i8, ptr addrspace(3) %grp.n164, i32 8384
  %bank.read21.addr = ptrtoint ptr addrspace(3) %lds.const.gep1244 to i32
  %bank.read21.shift = lshr i32 %bank.read21.addr, 6
  %bank.read21.mask = and i32 %bank.read21.shift, 8
  %bank.read21.xor = xor i32 %bank.read21.mask, %bank.read21.addr
  %bank.read21.ptr = inttoptr i32 %bank.read21.xor to ptr addrspace(3)
  %lds.const.gep1241 = getelementptr i8, ptr addrspace(3) %grp.n164, i32 12352
  %bank.read22.addr = ptrtoint ptr addrspace(3) %lds.const.gep1241 to i32
  %bank.read22.shift = lshr i32 %bank.read22.addr, 6
  %bank.read22.mask = and i32 %bank.read22.shift, 8
  %bank.read22.xor = xor i32 %bank.read22.mask, %bank.read22.addr
  %bank.read22.ptr = inttoptr i32 %bank.read22.xor to ptr addrspace(3)
  %lds.const.gep1245 = getelementptr i8, ptr addrspace(3) %grp.n164, i32 12480
  %bank.read23.addr = ptrtoint ptr addrspace(3) %lds.const.gep1245 to i32
  %bank.read23.shift = lshr i32 %bank.read23.addr, 6
  %bank.read23.mask = and i32 %bank.read23.shift, 8
  %bank.read23.xor = xor i32 %bank.read23.mask, %bank.read23.addr
  %bank.read23.ptr = inttoptr i32 %bank.read23.xor to ptr addrspace(3)
  %lds.const.gep1246 = getelementptr i8, ptr addrspace(3) %address.local559, i32 9216
  %bank.read24.addr = ptrtoint ptr addrspace(3) %lds.const.gep1246 to i32
  %bank.read24.shift = lshr i32 %bank.read24.addr, 6
  %bank.read24.mask = and i32 %bank.read24.shift, 8
  %bank.read24.xor = xor i32 %bank.read24.mask, %bank.read24.addr
  %bank.read24.ptr = inttoptr i32 %bank.read24.xor to ptr addrspace(3)
  %lds.const.gep1250 = getelementptr i8, ptr addrspace(3) %address.local559, i32 9344
  %bank.read25.addr = ptrtoint ptr addrspace(3) %lds.const.gep1250 to i32
  %bank.read25.shift = lshr i32 %bank.read25.addr, 6
  %bank.read25.mask = and i32 %bank.read25.shift, 8
  %bank.read25.xor = xor i32 %bank.read25.mask, %bank.read25.addr
  %bank.read25.ptr = inttoptr i32 %bank.read25.xor to ptr addrspace(3)
  %lds.const.gep1247 = getelementptr i8, ptr addrspace(3) %address.local559, i32 13312
  %bank.read26.addr = ptrtoint ptr addrspace(3) %lds.const.gep1247 to i32
  %bank.read26.shift = lshr i32 %bank.read26.addr, 6
  %bank.read26.mask = and i32 %bank.read26.shift, 8
  %bank.read26.xor = xor i32 %bank.read26.mask, %bank.read26.addr
  %bank.read26.ptr = inttoptr i32 %bank.read26.xor to ptr addrspace(3)
  %lds.const.gep1251 = getelementptr i8, ptr addrspace(3) %address.local559, i32 13440
  %bank.read27.addr = ptrtoint ptr addrspace(3) %lds.const.gep1251 to i32
  %bank.read27.shift = lshr i32 %bank.read27.addr, 6
  %bank.read27.mask = and i32 %bank.read27.shift, 8
  %bank.read27.xor = xor i32 %bank.read27.mask, %bank.read27.addr
  %bank.read27.ptr = inttoptr i32 %bank.read27.xor to ptr addrspace(3)
  %lds.const.gep1248 = getelementptr i8, ptr addrspace(3) %grp.n164, i32 9280
  %bank.read28.addr = ptrtoint ptr addrspace(3) %lds.const.gep1248 to i32
  %bank.read28.shift = lshr i32 %bank.read28.addr, 6
  %bank.read28.mask = and i32 %bank.read28.shift, 8
  %bank.read28.xor = xor i32 %bank.read28.mask, %bank.read28.addr
  %bank.read28.ptr = inttoptr i32 %bank.read28.xor to ptr addrspace(3)
  %lds.const.gep1252 = getelementptr i8, ptr addrspace(3) %grp.n164, i32 9408
  %bank.read29.addr = ptrtoint ptr addrspace(3) %lds.const.gep1252 to i32
  %bank.read29.shift = lshr i32 %bank.read29.addr, 6
  %bank.read29.mask = and i32 %bank.read29.shift, 8
  %bank.read29.xor = xor i32 %bank.read29.mask, %bank.read29.addr
  %bank.read29.ptr = inttoptr i32 %bank.read29.xor to ptr addrspace(3)
  %lds.const.gep1249 = getelementptr i8, ptr addrspace(3) %grp.n164, i32 13376
  %bank.read30.addr = ptrtoint ptr addrspace(3) %lds.const.gep1249 to i32
  %bank.read30.shift = lshr i32 %bank.read30.addr, 6
  %bank.read30.mask = and i32 %bank.read30.shift, 8
  %bank.read30.xor = xor i32 %bank.read30.mask, %bank.read30.addr
  %bank.read30.ptr = inttoptr i32 %bank.read30.xor to ptr addrspace(3)
  %lds.const.gep1253 = getelementptr i8, ptr addrspace(3) %grp.n164, i32 13504
  %bank.read31.addr = ptrtoint ptr addrspace(3) %lds.const.gep1253 to i32
  %bank.read31.shift = lshr i32 %bank.read31.addr, 6
  %bank.read31.mask = and i32 %bank.read31.shift, 8
  %bank.read31.xor = xor i32 %bank.read31.mask, %bank.read31.addr
  %bank.read31.ptr = inttoptr i32 %bank.read31.xor to ptr addrspace(3)
  %probe.value0 = load <4 x half>, ptr addrspace(3) %bank.read0.ptr, align 8
  %probe.value1 = load <4 x half>, ptr addrspace(3) %bank.read1.ptr, align 8
  %probe.value2 = load <4 x half>, ptr addrspace(3) %bank.read2.ptr, align 8
  %probe.value3 = load <4 x half>, ptr addrspace(3) %bank.read3.ptr, align 8
  %probe.value4 = load <4 x half>, ptr addrspace(3) %bank.read4.ptr, align 8
  %probe.value5 = load <4 x half>, ptr addrspace(3) %bank.read5.ptr, align 8
  %probe.value6 = load <4 x half>, ptr addrspace(3) %bank.read6.ptr, align 8
  %probe.value7 = load <4 x half>, ptr addrspace(3) %bank.read7.ptr, align 8
  %probe.value8 = load <4 x half>, ptr addrspace(3) %bank.read8.ptr, align 8
  %probe.value9 = load <4 x half>, ptr addrspace(3) %bank.read9.ptr, align 8
  %probe.value10 = load <4 x half>, ptr addrspace(3) %bank.read10.ptr, align 8
  %probe.value11 = load <4 x half>, ptr addrspace(3) %bank.read11.ptr, align 8
  %probe.value12 = load <4 x half>, ptr addrspace(3) %bank.read12.ptr, align 8
  %probe.value13 = load <4 x half>, ptr addrspace(3) %bank.read13.ptr, align 8
  %probe.value14 = load <4 x half>, ptr addrspace(3) %bank.read14.ptr, align 8
  %probe.value15 = load <4 x half>, ptr addrspace(3) %bank.read15.ptr, align 8
  %probe.value16 = load <4 x half>, ptr addrspace(3) %bank.read16.ptr, align 8
  %probe.value17 = load <4 x half>, ptr addrspace(3) %bank.read17.ptr, align 8
  %probe.value18 = load <4 x half>, ptr addrspace(3) %bank.read18.ptr, align 8
  %probe.value19 = load <4 x half>, ptr addrspace(3) %bank.read19.ptr, align 8
  %probe.value20 = load <4 x half>, ptr addrspace(3) %bank.read20.ptr, align 8
  %probe.value21 = load <4 x half>, ptr addrspace(3) %bank.read21.ptr, align 8
  %probe.value22 = load <4 x half>, ptr addrspace(3) %bank.read22.ptr, align 8
  %probe.value23 = load <4 x half>, ptr addrspace(3) %bank.read23.ptr, align 8
  %probe.value24 = load <4 x half>, ptr addrspace(3) %bank.read24.ptr, align 8
  %probe.value25 = load <4 x half>, ptr addrspace(3) %bank.read25.ptr, align 8
  %probe.value26 = load <4 x half>, ptr addrspace(3) %bank.read26.ptr, align 8
  %probe.value27 = load <4 x half>, ptr addrspace(3) %bank.read27.ptr, align 8
  %probe.value28 = load <4 x half>, ptr addrspace(3) %bank.read28.ptr, align 8
  %probe.value29 = load <4 x half>, ptr addrspace(3) %bank.read29.ptr, align 8
  %probe.value30 = load <4 x half>, ptr addrspace(3) %bank.read30.ptr, align 8
  %probe.value31 = load <4 x half>, ptr addrspace(3) %bank.read31.ptr, align 8
  %probe.off = shl nuw nsw i32 %grp.n25, 8
  %13 = zext nneg i32 %probe.off to i64
  %probe.base = getelementptr i8, ptr addrspace(1) %probe.out, i64 %13
  store <4 x half> %probe.value0, ptr addrspace(1) %probe.base, align 8
  %probe.dst1 = getelementptr i8, ptr addrspace(1) %probe.base, i64 8
  store <4 x half> %probe.value1, ptr addrspace(1) %probe.dst1, align 8
  %probe.dst2 = getelementptr i8, ptr addrspace(1) %probe.base, i64 16
  store <4 x half> %probe.value2, ptr addrspace(1) %probe.dst2, align 8
  %probe.dst3 = getelementptr i8, ptr addrspace(1) %probe.base, i64 24
  store <4 x half> %probe.value3, ptr addrspace(1) %probe.dst3, align 8
  %probe.dst4 = getelementptr i8, ptr addrspace(1) %probe.base, i64 32
  store <4 x half> %probe.value4, ptr addrspace(1) %probe.dst4, align 8
  %probe.dst5 = getelementptr i8, ptr addrspace(1) %probe.base, i64 40
  store <4 x half> %probe.value5, ptr addrspace(1) %probe.dst5, align 8
  %probe.dst6 = getelementptr i8, ptr addrspace(1) %probe.base, i64 48
  store <4 x half> %probe.value6, ptr addrspace(1) %probe.dst6, align 8
  %probe.dst7 = getelementptr i8, ptr addrspace(1) %probe.base, i64 56
  store <4 x half> %probe.value7, ptr addrspace(1) %probe.dst7, align 8
  %probe.dst8 = getelementptr i8, ptr addrspace(1) %probe.base, i64 64
  store <4 x half> %probe.value8, ptr addrspace(1) %probe.dst8, align 8
  %probe.dst9 = getelementptr i8, ptr addrspace(1) %probe.base, i64 72
  store <4 x half> %probe.value9, ptr addrspace(1) %probe.dst9, align 8
  %probe.dst10 = getelementptr i8, ptr addrspace(1) %probe.base, i64 80
  store <4 x half> %probe.value10, ptr addrspace(1) %probe.dst10, align 8
  %probe.dst11 = getelementptr i8, ptr addrspace(1) %probe.base, i64 88
  store <4 x half> %probe.value11, ptr addrspace(1) %probe.dst11, align 8
  %probe.dst12 = getelementptr i8, ptr addrspace(1) %probe.base, i64 96
  store <4 x half> %probe.value12, ptr addrspace(1) %probe.dst12, align 8
  %probe.dst13 = getelementptr i8, ptr addrspace(1) %probe.base, i64 104
  store <4 x half> %probe.value13, ptr addrspace(1) %probe.dst13, align 8
  %probe.dst14 = getelementptr i8, ptr addrspace(1) %probe.base, i64 112
  store <4 x half> %probe.value14, ptr addrspace(1) %probe.dst14, align 8
  %probe.dst15 = getelementptr i8, ptr addrspace(1) %probe.base, i64 120
  store <4 x half> %probe.value15, ptr addrspace(1) %probe.dst15, align 8
  %probe.dst16 = getelementptr i8, ptr addrspace(1) %probe.base, i64 128
  store <4 x half> %probe.value16, ptr addrspace(1) %probe.dst16, align 8
  %probe.dst17 = getelementptr i8, ptr addrspace(1) %probe.base, i64 136
  store <4 x half> %probe.value17, ptr addrspace(1) %probe.dst17, align 8
  %probe.dst18 = getelementptr i8, ptr addrspace(1) %probe.base, i64 144
  store <4 x half> %probe.value18, ptr addrspace(1) %probe.dst18, align 8
  %probe.dst19 = getelementptr i8, ptr addrspace(1) %probe.base, i64 152
  store <4 x half> %probe.value19, ptr addrspace(1) %probe.dst19, align 8
  %probe.dst20 = getelementptr i8, ptr addrspace(1) %probe.base, i64 160
  store <4 x half> %probe.value20, ptr addrspace(1) %probe.dst20, align 8
  %probe.dst21 = getelementptr i8, ptr addrspace(1) %probe.base, i64 168
  store <4 x half> %probe.value21, ptr addrspace(1) %probe.dst21, align 8
  %probe.dst22 = getelementptr i8, ptr addrspace(1) %probe.base, i64 176
  store <4 x half> %probe.value22, ptr addrspace(1) %probe.dst22, align 8
  %probe.dst23 = getelementptr i8, ptr addrspace(1) %probe.base, i64 184
  store <4 x half> %probe.value23, ptr addrspace(1) %probe.dst23, align 8
  %probe.dst24 = getelementptr i8, ptr addrspace(1) %probe.base, i64 192
  store <4 x half> %probe.value24, ptr addrspace(1) %probe.dst24, align 8
  %probe.dst25 = getelementptr i8, ptr addrspace(1) %probe.base, i64 200
  store <4 x half> %probe.value25, ptr addrspace(1) %probe.dst25, align 8
  %probe.dst26 = getelementptr i8, ptr addrspace(1) %probe.base, i64 208
  store <4 x half> %probe.value26, ptr addrspace(1) %probe.dst26, align 8
  %probe.dst27 = getelementptr i8, ptr addrspace(1) %probe.base, i64 216
  store <4 x half> %probe.value27, ptr addrspace(1) %probe.dst27, align 8
  %probe.dst28 = getelementptr i8, ptr addrspace(1) %probe.base, i64 224
  store <4 x half> %probe.value28, ptr addrspace(1) %probe.dst28, align 8
  %probe.dst29 = getelementptr i8, ptr addrspace(1) %probe.base, i64 232
  store <4 x half> %probe.value29, ptr addrspace(1) %probe.dst29, align 8
  %probe.dst30 = getelementptr i8, ptr addrspace(1) %probe.base, i64 240
  store <4 x half> %probe.value30, ptr addrspace(1) %probe.dst30, align 8
  %probe.dst31 = getelementptr i8, ptr addrspace(1) %probe.base, i64 248
  store <4 x half> %probe.value31, ptr addrspace(1) %probe.dst31, align 8
  ret void
}

attributes #0 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #1 = { nounwind memory(read, argmem: readwrite, inaccessiblemem: none, target_mem: none) "amdgpu-flat-work-group-size"="256,256" "amdgpu-no-cluster-id-x" "amdgpu-no-cluster-id-y" "amdgpu-no-cluster-id-z" "amdgpu-no-completion-action" "amdgpu-no-default-queue" "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-flat-scratch-init" "amdgpu-no-heap-ptr" "amdgpu-no-hostcall-ptr" "amdgpu-no-implicitarg-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-y" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" "amdgpu-no-wwm" }
attributes #2 = { convergent nounwind memory(none) }
