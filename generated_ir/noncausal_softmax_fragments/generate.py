#!/usr/bin/env python3
"""Apply the compiled scheduling pass to the best recorded non-causal artifact."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / 'generated_ir/noncausal_k_lds_prefetch_pass/mytest_final/run-20261030-171621-98fko2hz/online-vdb-k64-lds2/kernel.ll'
TOOL = ROOT / '3rd/deepgengraph/build/test/LegacyLLVMExportTest'
LLVM = Path('/data2/xsl/install/bin')


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def run(cmd):
    return subprocess.run(list(map(str, cmd)), text=True, capture_output=True, check=True)


def instructions(text):
    # Named input gives unchanged SSA identities even after instruction motion.
    # Ignore only metadata and attribute numbering, not operands/math flags.
    return [re.sub(r', ![\w.]+ !\d+', '', re.sub(r' #\d+', '', l.strip()))
            for l in text.splitlines() if l.startswith('  ') and not l.lstrip().startswith(';')
            and '!frisk.softmax_fragments' not in l]


def audit(base, candidate):
    a, b = instructions(base), instructions(candidate)
    assert Counter(a) == Counter(b), 'arithmetic/instruction multiset changed'
    def fixed(lines):
        return [l for l in lines if any(x in l for x in (
            'v_mmac_', 'buffer_load_', 's_waitcnt', 'fence ', '@llvm.amdgcn.s.barrier',
            '@llvm.hcu.ds.read.', '@llvm.amdgcn.sched.barrier', ' load ', 'store '))]
    assert fixed(a) == fixed(b), 'memory/sync/MMAC order changed'
    return dict(identical_instruction_multiset=True, identical_memory_sync_mmac_order=True)


def inspect(text):
    # Only the 16 P casts, not the final output casts.
    p = [(n, l) for n, l in enumerate(text.splitlines())
         if 'fptrunc <4 x float> %math.result' in l]
    lines = text.splitlines()
    mmas = [n for n,l in enumerate(lines) if 'v_mmac_f32_16x16x16_f16' in l and '%math.packed' in l]
    shifts = [n for n,l in enumerate(lines) if '= fsub <4 x float>' in l]
    assert len(p) == 16 and len(mmas) == 128 and len(shifts) == 16
    return dict(P_fragments=len(p), PV_mmacs=len(mmas),
                fragments_packed_before_first_PV=sum(n < mmas[0] for n,l in p),
                score_subtractions_before_first_pack=sum(n < p[0][0] for n in shifts),
                PV_mmacs_before_last_pack=sum(n < p[-1][0] for n in mmas),
                softmax_schedule_hints=sum('call void' in l and '!frisk.softmax_fragments' in l for l in lines))


def main():
    log = ROOT/'pmc_test.log'
    decoded = log.read_bytes().decode('utf-16')
    assert str(SOURCE.relative_to(ROOT)) in decoded.splitlines()[0]
    assert 'median=1.735664' in decoded.splitlines()[1]
    config = json.loads(SOURCE.with_name('config.json').read_text())
    assert config['source_input'].endswith('test_input_spidD_noncasual.mlir')
    assert config['parameters']['workgroup-group-size'] == '32'
    actual_prefetch = re.search(r'"frisk.k_lds_prefetch"="(\d+)"',SOURCE.read_text())[1]
    assert actual_prefetch == '8'
    shutil.copyfile(SOURCE, HERE/'control.ll')
    shutil.copyfile(SOURCE.with_name('config.json'), HERE/'baseline.config.json')
    shutil.copyfile(SOURCE.with_name('input.mlir'), HERE/'baseline.input.mlir')
    (HERE/'baseline.pmc.txt').write_text('\n'.join(decoded.splitlines()[:4])+'\n')
    run([LLVM/'opt', '-S', '-passes=instnamer', SOURCE, '-o', HERE/'control.named.ll'])
    base = (HERE/'control.named.ll').read_text()
    records = {}
    for name,mode in [('control','off'), ('softmax_fragment','fragment'), ('softmax_pv','pv')]:
        out = HERE/(name+'.ll')
        cmd = [TOOL, '--softmax-fragments='+mode, HERE/'control.named.ll']
        result = run(cmd)
        (HERE/(name+'.pass.log')).write_text(result.stderr)
        audit_result = audit(base, result.stdout)
        if mode != 'off':
            assert '16 fragments' in result.stderr
            raw = HERE/(name+'.raw.ll')
            raw.write_text(result.stdout)
            out.write_text(run([TOOL, '--llvm22-text', raw]).stdout)
        # control.ll is deliberately the exact measured artifact, not re-exported.
        out.with_suffix('.launch.json').write_text(json.dumps(dict(schema_version=1,
            kernel='Attn_p2', shape=[1,32,4096,128], grid=[32,32,1], block=[256,1,1],
            shared_mem=0, k_layout='bhsd', causal=False), indent=2)+'\n')
        run([LLVM/'llvm-as', out, '-o', '/dev/null'])
        run([LLVM/'opt', '-S', '-passes=default<O3>,verify', out, '-o', out.with_suffix('.o3.ll')])
        records[name] = dict(mode=mode, command=list(map(str,cmd)), sha256=digest(out),
            diagnostics=result.stderr.strip(), audit=audit_result,
            schedule=inspect(out.read_text()), O3_schedule=inspect(out.with_suffix('.o3.ll').read_text()),
            llvm_verified=True, O3_verified=True)
    assert records['softmax_fragment']['schedule']['score_subtractions_before_first_pack'] == 1
    assert records['softmax_pv']['schedule']['fragments_packed_before_first_PV'] == 1
    assert records['softmax_pv']['O3_schedule']['fragments_packed_before_first_PV'] == 1
    assert digest(HERE/'control.ll') == digest(SOURCE)
    (HERE/'manifest.json').write_text(json.dumps(dict(source=str(SOURCE), source_sha256=digest(SOURCE),
        pmc_sha256=digest(log), recorded_median_ms=1.735664, exporter_sha256=digest(TOOL),
        baseline_note='Actual IR and PMC both use K LDS prefetch=8; archived config says 2 and is stale',
        actual_k_lds_prefetch=int(actual_prefetch),
        variants=records, gpu_tested=False),indent=2)+'\n')
    for name,r in records.items(): print(name, r['schedule'])


if __name__ == '__main__': main()
