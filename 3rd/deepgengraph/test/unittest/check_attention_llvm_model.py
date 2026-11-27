#!/usr/bin/env python3
"""Check LLVM attention DMA, lazy-softmax decisions and packed rounding."""
import numpy as np
from check_attention_llvm_numerics import LLVMWorkgroupModel

def main():
    source = np.arange(2048,dtype=np.float16)
    module = '@shm_0 = addrspace(3) global [4096 x i8] undef\ndefine amdgpu_kernel void @test() {\n4:\n  ret void\n}\n'
    for width in (8,16):
        model = LLVMWorkgroupModel(module,source,source,source,threads=64)
        model.values['%lds'] = np.zeros(64,dtype=np.int64)
        model.values['%off'] = np.arange(64,dtype=np.int64)*width
        model.values['%res'] = np.tile(np.array([0,1,source.nbytes,0x20000],dtype=np.int64),(64,1))
        call = f'i32 asm sideeffect "s_mov_b32 $0, $1\\0A\\09buffer_load_dwordx{width//4} $2, $3, 0 offen lds", "={{m0}},s,v,s,~{{memory}}"(i32 %lds, i32 %off, <4 x i32> %res)'
        model.call(call)
        np.testing.assert_array_equal(model.mem[0][:64*width//2],source[:64*width//2])
        model.values['%off'][-1] = source.nbytes-width+1
        try:model.call(call)
        except AssertionError as error:assert 'buffer OOB' in str(error)
        else:raise AssertionError('out-of-bounds packet accepted')
        print('PASS DMA writes and packet bounds',width)
    model.values['%peak'] = np.tile(np.array([7.0,8.0,9.0,np.nan],np.float32),16)
    np.testing.assert_array_equal(model.operation('fcmp ogt float %peak, 8.0'),
                                  np.tile([False,False,True,False],16))
    print('PASS ordered lazy-softmax threshold comparison')
    # O3 vectorizes the causal mask comparisons; retain the element width
    # when interpreting unsigned vectors, including negative bit patterns.
    model.values['%coords'] = np.tile(np.array([-1, 0, 31, 32], np.int64), (64, 1))
    model.values['%limit'] = np.full((64, 4), 31, np.int64)
    for bits in (32, 64):
        np.testing.assert_array_equal(
            model.operation(f'icmp ugt <4 x i{bits}> %coords, %limit'),
            np.tile([True, False, False, True], (64, 1)))
    print('PASS unsigned vector causal-mask comparisons')
    model.values['%pred'] = np.arange(64)==63
    ballot = model.call('i64 @llvm.amdgcn.ballot.i64(i1 %pred)')
    np.testing.assert_array_equal(ballot,np.full(64,np.iinfo(np.int64).min))
    print('PASS wave ballot preserves lane 63')
    model.values['%lo'] = np.tile(np.array([1.0007,-1.0007,.3334,-.3334],np.float32),16)
    model.values['%hi'] = np.full(64,.5,np.float32)
    packed = model.call('i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %lo, float %hi)')
    halves = packed.astype(np.uint32).view(np.float16).reshape(64,2)
    np.testing.assert_array_equal(halves[:,0],np.tile(np.array([1,-1,.333251953125,-.333251953125],np.float16),16))
    np.testing.assert_array_equal(halves[:,1],np.full(64,.5,np.float16))
    print('PASS packed probabilities round toward zero')
    # A partially active load must not touch the five-element source for the
    # other 59 lanes; the reconverged PHI supplies zeros to those lanes.
    guarded = '''@shm_0 = addrspace(3) global [128 x i8] undef
define amdgpu_kernel void @test() {
4:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %inside = icmp ult i32 %tid, 5
  br i1 %inside, label %read, label %join
read:
  %ptr = getelementptr half, ptr addrspace(1) %1, i32 %tid
  %value = load half, ptr addrspace(1) %ptr
  br label %join
join:
  %result = phi half [ %value, %read ], [ 0.0, %4 ]
  %out = getelementptr half, ptr addrspace(1) %3, i32 %tid
  store half %result, ptr addrspace(1) %out
  ret void
}
'''
    q = np.zeros(64, dtype=np.float16)
    model = LLVMWorkgroupModel(guarded, q, np.arange(5, dtype=np.float16), q, threads=64)
    np.testing.assert_array_equal(model.run(), np.r_[np.arange(5), np.zeros(59)])
    print('PASS divergent guarded load, exact buffer bounds and reconverged PHI')

if __name__ == '__main__':main()
