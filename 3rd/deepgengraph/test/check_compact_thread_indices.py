#!/usr/bin/env python3
"""Execute original and compacted integer layouts over every legal workitem ID."""
import re
import subprocess
import sys
from pathlib import Path


def run(args, text):
    result = subprocess.run(list(map(str, args)), input=text, text=True,
                            capture_output=True, timeout=120)
    assert result.returncode == 0, result.stderr
    return result.stdout


def cpu_body(text, name):
    body = re.search(r'define i64 @index\(.*?\n\}', text, re.S)[0]
    ids = re.findall(r'^  (%[\w.]+) = call i32 @llvm.amdgcn.workitem.id.x\(\).*\n', body, re.M)
    body = re.sub(r'^  %[\w.]+ = call i32 @llvm.amdgcn.workitem.id.x\(\).*\n', '', body, flags=re.M)
    for value in ids:
        body = re.sub(re.escape(value) + r'(?![\w.])', '%input', body)
    body = re.sub(r' #\d+| !reqd_work_group_size !\d+|, ![\w.]+ !\d+', '', body)
    return body.replace('@index(', '@' + name + '(')


def main():
    tool = Path(sys.argv[1]).resolve()
    lli = Path('/data2/xsl/install/bin/lli')
    for count in (32, 64, 192, 256, 512, 1024):
        source = f'''target triple = "amdgcn-amd-amdhsa"
declare i32 @llvm.amdgcn.workitem.id.x()
define i64 @index(i32 %input) !reqd_work_group_size !0 {{
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %t = sext i32 %tid to i64
  %a = mul i64 %t, 18
  %b = sdiv i64 %a, 16
  %c = mul i64 %b, 16
  %d = sub i64 %a, %c
  %e = mul i64 %d, 1024
  %f = sdiv i64 %t, 16
  %g = srem i64 %f, 4
  %h = mul i64 %g, 8
  %out = add i64 %e, %h
  ret i64 %out
}}
!0 = !{{i32 {count}, i32 1, i32 1}}
'''
        output = run([tool, '--optimize-register-lifetimes', '-'], source)
        assert 'sdiv ' not in output and 'srem ' not in output
        cpu = cpu_body(source, 'before') + '\n' + cpu_body(output, 'after') + f'''
define i32 @main() {{
entry:
  br label %loop
loop:
  %i = phi i32 [0, %entry], [%next, %okay]
  %a = call i64 @before(i32 %i)
  %b = call i64 @after(i32 %i)
  %equal = icmp eq i64 %a, %b
  br i1 %equal, label %okay, label %fail
okay:
  %next = add i32 %i, 1
  %more = icmp ult i32 %next, {count}
  br i1 %more, label %loop, label %exit
fail:
  ret i32 1
exit:
  ret i32 0
}}
'''
        run([lli, '-'], cpu)
    print('PASS exhaustive CPU equality: 32/64/192/256/512/1024 workitems, signed div/mod layout inverses')


if __name__ == '__main__':
    main()
