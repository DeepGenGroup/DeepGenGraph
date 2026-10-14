#!/usr/bin/env python3
"""Check the zero-immediate invariant of a DMA-address-guard export.

This is a necessary codegen check, not GPU numerical validation. For arbitrary
unguarded assembly a nonzero immediate can be legal if the LDS base was also
adjusted; this checker deliberately does not attempt to prove compensation.
"""
import argparse
from pathlib import Path
import re


def dma_offsets(text, kernel):
    start = re.search(r'^' + re.escape(kernel) + r':[^\n]*\n', text, re.M)
    if start is None:
        raise ValueError(f'Kernel label not found: {kernel}')
    end = re.search(r'^\.Lfunc_end\d+:', text[start.end():], re.M)
    if end is None:
        raise ValueError('LLVM assembly function end marker not found')
    body = text[start.end():start.end() + end.start()]
    first_line = text[:start.end()].count('\n') + 1
    result = []
    for number, line in enumerate(body.splitlines(), first_line):
        instruction = line.split(';', 1)[0].strip()
        if (re.match(r'global_load_lds_\w+\b', instruction) or
                re.match(r'global_load_\w+\b', instruction) and re.search(r'\blds\b', instruction)):
            immediate = re.search(r'\boffset:\s*(-?(?:0x[\da-fA-F]+|\d+))', instruction)
            result.append((number, int(immediate[1], 0) if immediate else 0, instruction))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('assembly', type=Path)
    parser.add_argument('--kernel', default='Attn_p2')
    parser.add_argument('--expected-count', type=int, required=True)
    args = parser.parse_args()
    data = args.assembly.read_bytes()
    text = data.decode('utf-16' if data[:2] in (b'\xff\xfe', b'\xfe\xff') else 'utf-8')
    rows = dma_offsets(text, args.kernel)
    print(f'{args.kernel}: {len(rows)} DMA instructions, offsets={[offset for _, offset, _ in rows]}')
    bad = [row for row in rows if row[1]]
    for line, offset, instruction in bad:
        print(f'  line {line}: offset={offset}: {instruction}')
    if args.expected_count <= 0 or len(rows) != args.expected_count or bad:
        raise SystemExit('FAIL DMA address-guard invariant; inspect LDS base compensation if this is unguarded assembly')
    print('PASS expected DMA count and zero immediates; GPU correctness still requires execution')


if __name__ == '__main__':
    main()
