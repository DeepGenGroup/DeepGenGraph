#!/usr/bin/env python3
"""Export via compiled K-DMA options and compare with the measured DMA64 IR."""
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE.parent/'noncausal_v_double_buffer_banks'))
from generate import LLVM,digest
from verify import LazyAddresses, dma_args, vdb
spec=importlib.util.spec_from_file_location('dma64_model',HERE.parent/'noncausal_v_double_buffer_dma64/generate64.py')
helper=importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
TOOL=ROOT/'3rd/deepgengraph/build/test/LegacyLLVMExportTest'
SOURCE=HERE.parent/'noncausal_v_double_buffer_group32/v_double_buffer.ll'
MEASURED=HERE.parent/'noncausal_v_double_buffer_dma64/k_swizzle64_gep.ll'


class Model(helper.DMA64Model):
    def value(self, name, ty=None):
        constant=re.fullmatch(r'ptrtoint \((ptr addrspace\(3\) @[\w.]+) to i(?:32|64)\)',name)
        if constant: return self.typed(constant[1])
        return super().value(name,ty)


def generate():
    variants={}
    for name,mode in [('baseline','128'),('k_split64','64'),('k_swizzle64','64-swizzle')]:
        command=[str(TOOL),'--k-dma='+mode,'--k-dma-arg=1',str(SOURCE)]
        result=subprocess.run(command,text=True,capture_output=True,check=True)
        raw=HERE/(name+'.raw.ll'); raw.write_text(result.stdout)
        (HERE/(name+'.pass.log')).write_text(result.stderr)
        out=HERE/(name+'.ll')
        legalized=subprocess.run([TOOL,'--llvm22-text',raw],text=True,capture_output=True,check=True)
        out.write_text(legalized.stdout)
        out.with_suffix('.launch.json').write_bytes(SOURCE.with_suffix('.launch.json').read_bytes())
        subprocess.run([LLVM/'llvm-as',out,'-o','/dev/null'],check=True)
        subprocess.run([LLVM/'opt','-S','-passes=default<O3>,verify',out,'-o',out.with_suffix('.o3.ll')],check=True)
        variants[name]=dict(command=command,sha256=digest(out),mode=mode,llvm_and_O3_verified=True,
                            diagnostics=result.stderr.strip())
    (HERE/'manifest.json').write_text(json.dumps(dict(source=str(SOURCE),source_sha256=digest(SOURCE),
        measured_reference=str(MEASURED),measured_reference_sha256=digest(MEASURED),
        exporter_sha256=digest(TOOL),variants=variants,gpu_tested=False),indent=2)+'\n')


def audit_addresses():
    # Compare the compiler's recomputed source expression with the measured
    # experiment's delta expression, independently of SSA names and debug IDs.
    expected=LazyAddresses(MEASURED.read_text())
    actual=LazyAddresses((HERE/'k_swizzle64.ll').read_text())
    # Resource names may change, so select the emitted opcode, not a fixed name.
    def args(text):
        from check_gemm_pv_addresses import split_args
        return [split_args(re.search(r'"\((.*)\)',l)[1]) for l in text.splitlines() if 'buffer_load_dwordx2 ' in l]
    old_args=args(MEASURED.read_text()); new_args=args((HERE/'k_swizzle64.ll').read_text())
    assert len(old_args)==len(new_args)==16
    for head in range(32):
        for tile in range(32):
            expected.reset(head,0,tile);actual.reset(head,0,tile)
            for old,new in zip(old_args,new_args):
                for x,y in zip(map(expected.typed,old),map(actual.typed,new)):
                    assert np.array_equal(x,y)
    print('PASS all K DMA operands: 32 heads x 32 key tiles x 16 sites x 256 lanes',flush=True)


def main():
    generate();audit_addresses()
    records=[]
    for kind,head,query in [('random',0,0),('future_key_dominates',1,0),('max_changes',1,31)]:
        rng=np.random.default_rng(42+query)
        arrays=[rng.normal(size=(2,4096,128)).astype(np.float16) for _ in range(3)]
        if kind=='future_key_dominates':
            arrays[0].fill(1); arrays[1].fill(-1); arrays[1][:,-128:]=1
            arrays[2].fill(0); arrays[2][:,-128:]=1
        if kind=='max_changes':
            arrays[0].fill(1); arrays[1].fill(-16)
            arrays[1][:,:128]=.25; arrays[1][:,128:256]=.5
        expected=vdb.reference(*arrays,head,query,causal=False)
        paths={'measured':MEASURED,'baseline':HERE/'baseline.ll','split64':HERE/'k_split64.ll',
               'swizzle64':HERE/'k_swizzle64.ll'}
        if kind=='random':paths['swizzle64.o3']=HERE/'k_swizzle64.o3.ll'
        baseline=values=trace=None
        for name,path in paths.items():
            model=Model(re.sub(r'\bexact ','',path.read_text()),*arrays,threads=256,physical=(query,head))
            full=model.run().reshape(arrays[0].shape)
            actual=full[head,query*128:(query+1)*128]
            np.testing.assert_allclose(actual,expected,atol=.01,rtol=.01)
            assert not model.pending and model.vm_waits==260 and model.barriers==293 and model.mmac_count==8192
            assert model.v_reads_with_pending_dma==768
            assert len(model.q_elements)==16384 and set(model.q_elements.values())=={1}
            assert getattr(model,'dma64_count',0)==(0 if name=='baseline' else 512)
            if baseline is None:baseline=full.copy();values=model.lds_values.hexdigest();trace=(model.v_reads,model.v_dma)
            assert np.array_equal(full,baseline,equal_nan=True)
            assert values==model.lds_values.hexdigest() and trace==(model.v_reads,model.v_dma)
            assert np.isnan(full).sum()==full.size-16384
            records.append(dict(kind=kind,variant=name,bitwise_equal_to_measured=True,
                identical_LDS_read_values=True,V_schedule_unchanged=True,barrier_epoch_ownership=True,
                max_abs_error=float(np.max(np.abs(actual-expected)))))
            print('PASS numerical / delayed DMA / ownership',kind,name,flush=True)
    (HERE/'validation.json').write_text(json.dumps(dict(cases=records,all_K_DMA_operands_match=True,
        gpu_tested=False,gfx936_codegen_tested=False,independent_wave_timing_simulated=False),indent=2)+'\n')


if __name__=='__main__':main()
