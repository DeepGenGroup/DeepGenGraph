import sys
import re

def fix_llvm_ir_memory_attributes(input_file: str, output_file: str = None):
    if output_file is None:
        output_file = input_file

    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 常见的新旧 LLVM 内存属性映射规则
    replacements = {
        r'memory\(none\)': 'readnone',
        r'memory\(read\)': 'readonly',
        r'memory\(write\)': 'writeonly',
        r'memory\(argmem:\s*readwrite\)': 'argmemonly',
        r'memory\(argmem:\s*read\)': 'argmemonly readonly',
        r'memory\(argmem:\s*write\)': 'argmemonly writeonly',
        r'memory\(inaccessiblemem:\s*readwrite\)': 'inaccessiblememonly',
        r'memory\(inaccessiblemem:\s*read\)': 'inaccessiblememonly readonly',
        r'memory\(inaccessiblemem:\s*write\)': 'inaccessiblememonly writeonly',
        r'memory\(argmem:\s*readwrite,\s*inaccessiblemem:\s*readwrite\)': 'inaccessiblemem_or_argmemonly',
    }

    modified_content = content
    replaced_count = 0

    for pattern, repl in replacements.items():
        modified_content, count = re.subn(pattern, repl, modified_content)
        replaced_count += count

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(modified_content)

    print(f"[Done] 处理完成，共替换了 {replaced_count} 处内存属性。输出文件: {output_file}")

if __name__ == "__main__":
    # 支持命令行参数，如：python fix_ir.py final.ll
    target_file = sys.argv[1] if len(sys.argv) > 1 else "final.ll"
    fix_llvm_ir_memory_attributes(target_file)