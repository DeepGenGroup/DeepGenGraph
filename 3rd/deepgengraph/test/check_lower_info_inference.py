#!/usr/bin/env python3
"""Check deferred LowerInfo inference and terminal failure diagnostics.

Usage: python3 check_lower_info_inference.py [path/to/LowerInfoInferenceTest]
Run against a Debug build to catch premature inference assertions.
"""
import resource
import subprocess
import sys
from pathlib import Path


def main():
    binary = (Path(sys.argv[1]) if len(sys.argv) > 1 else
              Path(__file__).resolve().parent.parent /
              "build/test/LowerInfoInferenceTest").resolve()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    cases = [([], "LowerInfo deferred broadcast inference tests passed"),
             (["--unresolved"], "unresolved ops remain"),
             (["--no-anchor"], "No direct infer anchor op")]
    for args, expected in cases:
        result = subprocess.run([str(binary), *args], text=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, timeout=30)
        if args:
            assert result.returncode != 0, result.stdout
            assert "LLVM ERROR: LowerInfo infer failed" in result.stdout, result.stdout
        else:
            assert result.returncode == 0, result.stdout
        assert expected in result.stdout, result.stdout
        assert "inferArithmeticOp error" not in result.stdout, result.stdout
        assert "inferError 6" not in result.stdout, result.stdout
        print("PASS", args[0] if args else "deferred broadcast")


if __name__ == "__main__":
    main()
