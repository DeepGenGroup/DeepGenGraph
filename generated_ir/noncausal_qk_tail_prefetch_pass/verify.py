#!/usr/bin/env python3
"""Run the established delayed-DMA model on the compiler pass outputs."""
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('pass_tail_validation',
    HERE.parent / 'noncausal_qk_tail_prefetch/verify.py')
validation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validation)
validation.HERE = HERE

if __name__ == '__main__':
    validation.main()
