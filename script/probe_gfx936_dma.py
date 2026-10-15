#!/usr/bin/env python3
"""Compile-only DMA capability probes for the exact target DTK installation.

No kernel is executed. Successful instruction selection alone does not prove
LDS lane layout, completion semantics, GPU correctness or performance.
"""
import argparse
import json
from pathlib import Path
import re
import resource
import shlex
import subprocess


CASES = [
    ('amdgcn_async_4', 'llvm.amdgcn.global.load.async.lds', 4, [4, 0, 0]),
    ('amdgcn_async_8', 'llvm.amdgcn.global.load.async.lds', 8, [8, 0, 0]),
    ('amdgcn_async_16', 'llvm.amdgcn.global.load.async.lds', 16, [16, 0, 0]),
    ('hcu_dword', 'llvm.hcu.global.load.dword.lds', 4, [0, 0, 0]),
    ('hcu_dwordx2', 'llvm.hcu.global.load.dwordx2.lds', 8, [0, 0, 0]),
    ('hcu_dwordx4', 'llvm.hcu.global.load.dwordx4.lds', 16, [0, 0, 0]),
    # These use a different completion path; they are NOT drop-in replacements
    # for asyncmark copies. Probe only whether the backend selects an opcode.
    ('amdgcn_legacy_4', 'llvm.amdgcn.global.load.lds', 4, [4, 0, 0]),
    ('amdgcn_legacy_16', 'llvm.amdgcn.global.load.lds', 16, [16, 0, 0]),
]

BUFFER_CASES = [
    ('raw_async_4', 'llvm.amdgcn.raw.buffer.load.async.lds', 4, []),
    ('raw_async_16', 'llvm.amdgcn.raw.buffer.load.async.lds', 16, []),
    ('hcu_raw_4', 'llvm.hcu.raw.buffer.load.lds', 4, []),
    ('hcu_raw_8', 'llvm.hcu.raw.buffer.load.lds', 8, []),
    ('hcu_raw_16', 'llvm.hcu.raw.buffer.load.lds', 16, []),
]


def source(intrinsic, width, controls, markers=False):
    if '.buffer.' in intrinsic:
        # Descriptor is an opaque uniform kernel argument: selection only.
        # This probe does not assert descriptor encoding or completion semantics.
        return f'''target triple = "amdgcn-amd-amdhsa"
@lds = addrspace(3) global [1024 x i8] undef, align 16
declare i32 @llvm.amdgcn.workitem.id.x()
declare void @{intrinsic}(<4 x i32>, ptr addrspace(3), i32 immarg, i32, i32, i32 immarg, i32 immarg)
define amdgpu_kernel void @dma_probe(<4 x i32> %rsrc) #0 {{
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %offset = mul i32 %tid, {width}
  call void @{intrinsic}(<4 x i32> %rsrc, ptr addrspace(3) @lds, i32 {width}, i32 %offset, i32 0, i32 0, i32 0)
  ret void
}}
attributes #0 = {{ "amdgpu-flat-work-group-size"="64,64" }}
'''
    control_text = ', '.join(f'i32 {v}' for v in controls)
    completion = '''  call void @llvm.amdgcn.asyncmark()
  call void @llvm.amdgcn.wait.asyncmark(i16 0)
''' if markers else ''
    return f'''; Compile-only probe; do not execute this kernel.
target triple = "amdgcn-amd-amdhsa"
@lds = addrspace(3) global [1024 x i8] undef, align 16
declare i32 @llvm.amdgcn.workitem.id.x()
declare void @{intrinsic}(ptr addrspace(1), ptr addrspace(3), i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.amdgcn.asyncmark()
declare void @llvm.amdgcn.wait.asyncmark(i16 immarg)
define amdgpu_kernel void @dma_probe(ptr addrspace(1) %src) #0 {{
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %index = mul i32 %tid, {width}
  %index64 = zext i32 %index to i64
  %p = getelementptr i8, ptr addrspace(1) %src, i64 %index64
  %bits = ptrtoint ptr addrspace(1) %p to i64
  %opaque = call i64 asm sideeffect "", "=v,0"(i64 %bits)
  %global = inttoptr i64 %opaque to ptr addrspace(1)
  call void @{intrinsic}(ptr addrspace(1) %global, ptr addrspace(3) @lds, {control_text})
{completion}  ret void
}}
attributes #0 = {{ "amdgpu-flat-work-group-size"="64,64" }}
'''


def invoke(command):
    try:
        result = subprocess.run(list(map(str, command)), capture_output=True,
                                text=True, timeout=60)
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired as error:
        return 124, '', 'TIMEOUT: ' + str(error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--llvm-bin', type=Path, required=True)
    parser.add_argument('--arch', default='gfx936')
    parser.add_argument('--mattr', default='', help='Explicit llc target features, e.g. +glb2lds-dword-x2x4; compile-only, not hardware validation')
    parser.add_argument('--out', type=Path, default=Path('dma_probe_results'))
    parser.add_argument('--emit-only', action='store_true', help='Write probes without running llc')
    parser.add_argument('--buffer-only', action='store_true', help='Probe the separate raw-buffer-to-LDS path')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    llc = args.llvm_bin.resolve() / 'llc'
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    cases = CASES + [('amdgcn_async_4_markers', CASES[0][1], 4, [4, 0, 0]),
                     ('hcu_dwordx4_markers', CASES[5][1], 16, [0, 0, 0])]
    if args.buffer_only:
        cases = BUFFER_CASES
    feature_args = [f'-mattr={args.mattr}'] if args.mattr else []
    report = {'llc': str(llc), 'arch': args.arch, 'mattr': args.mattr,
              'compile_only': True, 'cases': {}}
    if not args.emit_only:
        if not llc.is_file():
            parser.error(f'llc not found: {llc}')
        code, stdout, stderr = invoke([llc, '--version'])
        (args.out / 'llc_version.txt').write_text(stdout + stderr)
        report['version_exit_code'] = code
    for name, intrinsic, width, controls in cases:
        ir = (args.out / f'{name}.ll').resolve()
        assembly = (args.out / f'{name}.s').resolve()
        ir.write_text(source(intrinsic, width, controls, name.endswith('_markers')))
        if args.emit_only:
            continue
        command = [str(llc), '-mtriple=amdgcn-amd-amdhsa', f'-mcpu={args.arch}',
                   *feature_args, '-O3', '-filetype=asm', '-amdhsa-code-object-version=4', str(ir), '-o', str(assembly)]
        code, stdout, stderr = invoke(command)
        (args.out / f'{name}.log').write_text(shlex.join(command) + '\n' + stdout + stderr)
        text = assembly.read_text() if code == 0 and assembly.exists() else ''
        # A compiler lacking HCU definitions may emit an ordinary external call.
        # Treat that as unsupported, even when llc returns success.
        instructions = [line.strip() for line in text.splitlines()
                        if re.match(r'\s*(?:global|buffer)_load\w*', line) and
                        ('_lds' in line or re.search(r'\blds\b', line))]
        unknown_feature = any('not a recognized feature' in line for line in stderr.splitlines())
        external_intrinsic = any(name in text for name in
                                 (intrinsic, 'llvm.amdgcn.asyncmark', 'llvm.amdgcn.wait.asyncmark'))
        selected = code == 0 and bool(instructions) and not external_intrinsic and not unknown_feature
        errors = [line for line in stderr.splitlines()
                  if any(word in line for word in ('LLVM ERROR:', 'error:', 'not support', 'not a recognized'))]
        if code == 0 and external_intrinsic:
            errors.append('Intrinsic remains an external symbol; no intrinsic lowering')
        elif code == 0 and not instructions:
            errors.append('No global/buffer-to-LDS instruction found; inspect assembly')
        report['cases'][name] = {'command': command, 'exit_code': code,
                                'selected_lds_instruction': selected,
                                'instructions': instructions, 'errors': errors[:3]}
        print(f'{name:28s} {"SELECTED" if selected else "NOT SUPPORTED"} '
              f'{instructions[0] if selected else (errors[0] if errors else "see log")}', flush=True)
    if not args.emit_only:
        probe = (args.out / f'{cases[0][0]}.ll').resolve()
        _, stdout, stderr = invoke([llc, '-mtriple=amdgcn-amd-amdhsa', f'-mcpu={args.arch}',
                                   '-mattr=help', probe, '-o', '/dev/null'])
        (args.out / 'target_features.txt').write_text(stdout + stderr)
        (args.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        print(f'Results: {args.out / "report.json"}')
    else:
        print(f'Wrote {len(cases)} compile-only probes to {args.out}')


if __name__ == '__main__':
    main()
