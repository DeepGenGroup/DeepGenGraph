#!/usr/bin/env python3
"""Replay the exact pass_pk_rtz generation without overwriting any IR artifact."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LLVM = Path('/data2/xsl/install/bin')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    manifest = json.loads((HERE/'manifest.json').read_text())
    record = manifest['variants']['pass_pk_rtz']
    command = record['command']
    tool, source = Path(command[0]), Path(command[-1])
    raw = HERE/'pass_pk_rtz.raw.ll'
    compatible = HERE/'pass_pk_rtz.ll'
    final = HERE/'pass_pk_rtz.o3.ll'
    original = {p: p.read_bytes() for p in (source, raw, compatible, final)}
    assert sha(original[source]) == record['source_sha256']
    assert sha(original[compatible]) == record['sha256']
    assert sha(original[final]) == record['o3_sha256']
    current_binary_sha = sha(tool.read_bytes())
    sections = [
        '; pass_pk_rtz.o3.ll lowering log — exact generation replay\n'
        '; 起点是已有 LLVM IR lazy_t8.ll；本日志不包含更早的 MLIR lowering。\n'
        '; 当前 MyTest 全流水线验证走的是另一条路径，其日志不能代表本文件。\n'
        '; 实际重放：SoftmaxCvtPack -> LLVM22 文本兼容处理 -> LLVM default<O3>,verify。\n'
        '; 所有阶段均与保存的产物逐字节核对，不覆盖原 IR。\n'
        f'; Target artifact: {final}\n'
        f'; Expected SHA256: {record["o3_sha256"]}\n'
        f'; Historical Pass binary SHA256: {manifest["pass_binary_sha256"]}\n'
        f'; Replay Pass binary SHA256: {current_binary_sha}\n'
        '; 工具二进制可能已重建；重放结果必须逐阶段与历史 IR 完全相同。\n',
        '---------------- input lazy_t8 LLVM IR ----------------\n'
        f'; Source: {source}\n; SHA256: {sha(original[source])}\n'+original[source].decode(),
    ]

    def replay(title, cmd, expected):
        result = subprocess.run(list(map(str, cmd)), capture_output=True, check=True)
        assert result.stdout == original[expected], ('Replay differs from saved artifact', title)
        sections.append(f'---------------- {title} ----------------\n'
                        f'; Command: {shlex.join(list(map(str, cmd)))}\n'
                        f'; SHA256: {sha(result.stdout)}\n'
                        '; Diagnostics / pass execution trace:\n'+
                        ''.join('; '+line+'\n' for line in result.stderr.decode().splitlines())+
                        '; Complete output IR:\n'+result.stdout.decode())

    replay('llvm softmaxCvtPack (16 fragments / 32 PKRTZ pairs)', command, raw)
    replay('LLVM22 text compatibility', [tool, '--llvm22-text', raw], compatible)
    replay('LLVM default<O3> and verifier',
           [LLVM/'opt', '-S', '-passes=default<O3>,verify', '-debug-pass-manager',
            compatible, '-o', '-'], final)
    for path, before in original.items():
        assert path.read_bytes() == before, ('Source/artifact was modified', path)
    sections.append('; VERIFIED: every replayed stage is byte-identical to its saved artifact.\n'
                    f'; Final SHA256: {sha(original[final])}\n')
    log = HERE/'pass_pk_rtz.lowering.log'
    log.write_text('\n'.join(sections))
    print(log)
    print('Verified final SHA256:', sha(original[final]))


if __name__ == '__main__':
    main()
