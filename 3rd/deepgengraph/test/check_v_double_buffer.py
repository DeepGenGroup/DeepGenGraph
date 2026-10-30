#!/usr/bin/env python3
"""V double-buffer legality, generality, idempotence and CLI regressions."""
import argparse
from pathlib import Path
import re
import subprocess
import tempfile


DMA = 'asm sideeffect "s_mov_b32 $0, $1\\0A\\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"'
WAIT = '  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"()'
SYNC = ['  fence syncscope("workgroup") release',
        '  call void @llvm.amdgcn.s.barrier()',
        '  fence syncscope("workgroup") acquire']


def fixture(threads=64, packets=1, stages=3, base=0, capacity=None):
    size = threads*16*packets
    capacity = capacity if capacity is not None else max(base+size,2*size)
    lines = [f'@space = addrspace(3) global [{capacity} x i8] undef, align 16',
             'declare i32 @llvm.amdgcn.workitem.id.x()',
             'declare i32 @llvm.amdgcn.readfirstlane.i32(i32)',
             'declare void @llvm.amdgcn.s.barrier()',
             'declare void @unknown()',
             'declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32)',
             'define amdgpu_kernel void @generic(ptr addrspace(1) %out, <4 x i32> %resource, <4 x half> %a) !reqd_work_group_size !0 {',
             'entry:', '  %tid = call i32 @llvm.amdgcn.workitem.id.x()',
             '  %offset = mul i32 %tid, 16',
             '  %local = add i32 %offset, '+str(base),
             '  %readptr = getelementptr i8, ptr addrspace(3) @space, i32 %local',
             '  %address = ptrtoint ptr addrspace(3) %readptr to i32',
             '  %leader = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address)']
    for packet in range(packets):
        lines += [f'  %dma{packet} = add i32 %leader, {packet*threads*16}',
                  f'  %ptr{packet} = getelementptr i8, ptr addrspace(3) %readptr, i32 {packet*threads*16}']
    def issue(stage):
        return [f'  %copy{stage}_{packet} = call i32 {DMA}(i32 %dma{packet}, i32 {stage*size+packet*threads*16}, <4 x i32> %resource)'
                for packet in range(packets)]
    lines += issue(0)
    for stage in range(stages):
        lines += [WAIT]+SYNC
        for packet in range(packets):
            lines += [f'  %r{stage}_{packet} = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %ptr{packet}, i32 0)',
                      f'  %b{stage}_{packet} = bitcast <4 x i32> %r{stage}_{packet} to <8 x half>',
                      f'  %h{stage}_{packet} = shufflevector <8 x half> %b{stage}_{packet}, <8 x half> poison, <4 x i32> <i32 0,i32 1,i32 2,i32 3>']
        if stage+1 < stages:
            lines += SYNC+issue(stage+1)
        acc = 'zeroinitializer' if stage == 0 else f'%m{stage-1}'
        lines += [f'  %m{stage} = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", "=v,v,v,0"(<4 x half> %a, <4 x half> %h{stage}_0, <4 x float> {acc}) #0']
    lines += SYNC+[f'  store <4 x float> %m{stages-1}, ptr addrspace(1) %out',
                   '  ret void', '}', 'attributes #0 = { convergent nounwind willreturn memory(none) }',
                   f'!0 = !{{i32 {threads}, i32 1, i32 1}}']
    return '\n'.join(lines)+'\n'


def instructions(text):
    return [re.sub(r', ![\w.]+ !\d+', '', re.sub(r' #\d+', '', l.strip()))
            for l in text.splitlines() if l.startswith('  ') and not l.lstrip().startswith(';')]


def with_loop(text):
    start = text.index('  %copy0_0 =')
    text = text[:start]+'''  br label %header
header:
  %iteration = phi i32 [0, %entry], [%next, %body]
  %more = icmp ult i32 %iteration, 3
  br i1 %more, label %body, label %exit
body:
'''+text[start:]
    return text.replace('  ret void','''  %next = add i32 %iteration, 1
  br label %header
exit:
  ret void''')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exporter', type=Path, default=Path(__file__).resolve().parents[1]/'build/test/LegacyLLVMExportTest')
    args = p.parse_args()
    cases = 0
    with tempfile.TemporaryDirectory(prefix='v-double-buffer-') as tmp:
        path = Path(tmp)/'input.ll'
        def run(text, enabled=1):
            path.write_text(text)
            result = subprocess.run([str(args.exporter), f'--v-double-buffer={enabled}', str(path)], text=True, capture_output=True)
            assert result.returncode == 0, result.stderr
            return result.stdout, result.stderr
        for threads,packets,stages,base in [(64,1,2,0),(128,2,3,0),(256,2,4,0),(64,4,5,4096),(128,1,2,2048)]:
            text = fixture(threads,packets,stages,base)
            out,log = run(text)
            assert f'{stages} slices, slot-bytes={threads*packets*16}' in log, log
            assert out.count('call void @llvm.amdgcn.s.barrier()') == stages+1
            assert out.count('s_waitcnt vmcnt(0)') == stages
            assert len(re.findall(r'!frisk.v_double_buffer !',out)) == stages*packets
            assert instructions(run(out)[0]) == instructions(out), 'idempotence'
            before = [l for l in instructions(run(text,0)[0]) if 'v_mmac_' in l]
            after = [l for l in instructions(out) if 'v_mmac_' in l]
            assert before == after
            cases += 1
        base = fixture()
        renamed = re.sub(r'%([A-Za-z]\w*)',r'%renamed_\1',base).replace('@space','@renamed_storage').replace('@generic','@renamed_kernel')
        assert '!frisk.v_double_buffer' in run(renamed)[0]
        cases += 1
        fresh = []
        for line in base.splitlines():
            if 'buffer_load_dwordx4' in line:
                name = re.search(r'%copy(\w+) =',line)[1]
                fresh.append(f'  %descriptor{name} = shufflevector <4 x i32> %resource, <4 x i32> poison, <4 x i32> <i32 0,i32 1,i32 2,i32 3>')
                line = line.replace('<4 x i32> %resource)',f'<4 x i32> %descriptor{name})')
            fresh.append(line)
        fresh = with_loop('\n'.join(fresh)+'\n')
        assert '!frisk.v_double_buffer' in run(fresh)[0]
        cases += 1
        loop = with_loop(base)
        bad = {
            'no_capacity': fixture(capacity=1024),
            'nonuniform_dma': base.replace('i32 %dma0, i32','i32 %address, i32'),
            'unknown_call': base.replace('  %r0_0 =','  call void @unknown()\n  %r0_0 ='),
            'live_other_lds': base.replace('  %r0_0 =','  %live = load <4 x half>, ptr addrspace(3) %readptr\n  %r0_0 ='),
            'late_lds_reader': base.replace('  ret void','  %late = load <4 x half>, ptr addrspace(3) %readptr\n  ret void'),
            'volatile_store': base.replace('  %r0_0 =','  store volatile i32 0, ptr addrspace(3) %readptr\n  %r0_0 ='),
            'read_out_of_slice': base.replace('ptr addrspace(3) %ptr0, i32 0)', 'ptr addrspace(3) %ptr0, i32 2048)'),
            'missing_ready_wait': base.replace(WAIT+'\n','',1),
            'missing_reader_barrier': base.replace('\n'.join(SYNC)+'\n  %copy1_0', '  %copy1_0',1),
            'wave32': base.replace(' !reqd_work_group_size',' "target-features"="+wavefrontsize32" !reqd_work_group_size'),
            'two_dimensional': base.replace('!{i32 64, i32 1, i32 1}', '!{i32 64, i32 2, i32 1}'),
            'unfixed_workgroup': base.replace(' !reqd_work_group_size !0',''),
            'other_dma_source': base.replace('<4 x i32> %resource)\n  %m0', '<4 x i32> zeroinitializer)\n  %m0'),
            'unknown_mmac_effect': base.replace('float> asm "v_mmac', 'float> asm sideeffect "v_mmac'),
            'divergent_control': loop.replace('icmp ult i32 %iteration, 3','icmp ult i32 %tid, 32'),
            'loop_carried_slot': loop.replace('body:\n','body:\n  %oldptr = getelementptr i8, ptr addrspace(3) @space, i32 1024\n  %old = load <4 x half>, ptr addrspace(3) %oldptr\n'),
        }
        for name,text in bad.items():
            assert text != base or name == 'no_capacity',name
            out,log = run(text)
            assert '!frisk.v_double_buffer' not in out,(name,log)
            assert instructions(out)==instructions(run(text,0)[0]),name
            cases += 1
        for value in ('2','-1','true','4294967296',''):
            path.write_text(base)
            result = subprocess.run([str(args.exporter),f'--v-double-buffer={value}',str(path)],capture_output=True)
            assert result.returncode != 0,value
            cases += 1
    print(f'PASS {cases} V double-buffer cases (sizes, packet/stage counts, reverse slot, renaming, safety, idempotence, CLI)')


if __name__ == '__main__':
    main()
