#!/usr/bin/env python3
"""Export an unchanged baseline and a generic LDS-read canonicalization candidate."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parent.parent


def digest(data):
    return hashlib.sha256(data).hexdigest()


def metrics(text):
    loads = re.findall(r'^\s+.*= load [^\n]*ptr addrspace\(3\)[^\n]*', text, re.M)
    alignments = {}
    for line in loads:
        alignment = re.search(r', align (\d+)', line)
        key = alignment[1] if alignment else 'unspecified'
        alignments[key] = alignments.get(key, 0) + 1
    return dict(lds_loads=len(loads), lds_load_alignments=alignments,
                dma128_sites=text.count('buffer_load_dwordx4 '),
                block_barriers=text.count('call void @llvm.amdgcn.s.barrier('),
                vm_wait_sites=text.count('s_waitcnt vmcnt(0)'),
                mmac_sites=text.count('v_mmac_f32_16x16x16_f16 '))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path, help='Optional launch JSON copied for both variants')
    parser.add_argument('--export-tool', type=Path,
                        default=ROOT/'3rd/deepgengraph/build/test/LegacyLLVMExportTest')
    parser.add_argument('--llvm-bin', type=Path, default=Path('/data2/xsl/install/bin'))
    args = parser.parse_args()
    source = args.input.read_bytes()
    config = json.loads(args.config.read_text()) if args.config else None
    transformed = subprocess.run(
        [str(args.export_tool), '--canonicalize-lds-geps-only', '-'],
        input=source.decode(), text=True, capture_output=True, check=True)
    # LLVM's decimal FP printer differs by version. Preserve the represented
    # FP bits using the project's existing LLVM-22 text compatibility helper.
    candidate = subprocess.run([str(args.export_tool), '--llvm22-text', '-'],
        input=transformed.stdout, text=True, capture_output=True, check=True).stdout
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    baseline_path, candidate_path = out/'baseline.ll', out/'optimized.ll'
    baseline_path.write_bytes(source)
    candidate_path.write_text(candidate)
    for path in (baseline_path, candidate_path):
        subprocess.run([str(args.llvm_bin/'llvm-as'), str(path), '-o', '/dev/null'], check=True)
        subprocess.run([str(args.llvm_bin/'opt'), '-passes=default<O3>,verify',
                        str(path), '-disable-output'], check=True)
    if config is not None:
        for name in ('baseline', 'optimized'):
            (out/f'{name}.launch.json').write_text(json.dumps(config, indent=2)+'\n')
    manifest = dict(source=str(args.input.resolve()), source_sha256=digest(source),
        baseline_sha256=digest(baseline_path.read_bytes()),
        optimized_sha256=digest(candidate_path.read_bytes()),
        pass_name='CanonicalizeLDSGEPsPass',
        export_tool_sha256=digest(args.export_tool.read_bytes()),
        pass_source_sha256=digest((ROOT/'3rd/deepgengraph/lib/Conversion/ConvertToLLVM/CanonicalizeLDSGEPs.cpp').read_bytes()),
        diagnostics=transformed.stderr.strip(),
        baseline=metrics(source.decode()), optimized=metrics(candidate),
        llvm_assemble_and_O3_verify=True, gpu_tested=False,
        target_gfx936_codegen_tested=False)
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
