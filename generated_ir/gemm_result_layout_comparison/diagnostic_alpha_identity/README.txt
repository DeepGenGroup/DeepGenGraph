3.1 safe-softmax failure diagnosis

Root: Exp2OpTiling, FriskBaseToTheadlevelIR_plan2.cpp:2000-2012.
It materializes the operand using inputInfo, computes exp2 without changing
lane/register ownership, then labels the result with output info.
For alpha, input and output layouts have equal vector<2x1xf32> shape but
different row ownership. A shape-preserving operation cannot relabel them.

Evidence in ../layout_3_1/thread.mlir:728-767:
alpha is computed from rows 16*wave + lane%16 + 64*register.
The from_threadTile at line 763 instead labels it as
32*wave + lane%16 + 16*register (warp_inst_unroll=[2,1]).
Finalization materializes the supposed layout change through LDS. Its stores
therefore attach the alpha values to incorrect query rows, and consumers use
the wrong alpha for both L_old and O_old.

Example: wave=0, lane=0 holds alpha for rows [0,64], labelled as [0,16].
A consumer for row 16 consequently obtains alpha for row 64. Initial state is
L=O=0, so the error is hidden on the first key tile; row-uniform alpha also
hides it, explaining the constant overflow/underflow/max_changes test passes.

Diagnostic: replace only the ten alpha reload uses by original %1228/%1231.
The QK GEMM shared conversion, row maxima, exponentials, PV and all other
instructions remain unchanged. Unused LDS alpha stores/barriers remain too.
generate.py is bound to the exact failing source hash.

Validation: opt default<O3>,verify and all six actual-LLVM CPU execution cases
pass. Random query 1: max error 0.4730803 -> 0.0007240294.
Random query 31: max error 0.05241151 -> 0.0000958800.
See validation.json and validation.log. No GPU testing performed.

General repair direction: materialize exp2 input in the result layout before
computation (as FriskBinaryOpTiling already does), or keep the input layout on
the result and materialize any needed conversion explicitly. Never equate
vector shape with logical row ownership. Compiler source is not changed here.

Reproduce from repository root:
  python3 generated_ir/gemm_result_layout_comparison/diagnostic_alpha_identity/generate.py
  OPENBLAS_NUM_THREADS=1 python3 generated_ir/gemm_result_layout_comparison/validate.py diagnostic_alpha_identity
