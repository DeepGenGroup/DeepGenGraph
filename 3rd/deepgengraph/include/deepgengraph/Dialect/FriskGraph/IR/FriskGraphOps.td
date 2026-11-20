#ifndef FRISK_OPS
#define FRISK_OPS

include "deepgengraph/Dialect/Frisk/IR/FriskDialect.td"
include "deepgengraph/Dialect/Frisk/IR/FriskEnums.td"
include "deepgengraph/Dialect/Frisk/IR/FriskAttributes.td"
include "deepgengraph/Dialect/Frisk/IR/FriskTypes.td"
include "deepgengraph/Dialect/Frisk/IR/FriskInterfaces.td"

include "mlir/IR/OpBase.td"
// include "mlir/IR/AttrTypeBase.td"
// include "mlir/IR/EnumAttr.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/SymbolInterfaces.td" // SymbolUserOpInterface
include "mlir/Interfaces/CallInterfaces.td" // CallOpInterface
include "mlir/IR/OpAsmInterface.td" // OpAsmOpInterface
include "mlir/Interfaces/FunctionInterfaces.td" // FunctionOpInterface
include "mlir/Interfaces/SideEffectInterfaces.td" // Pure
include "mlir/Interfaces/ControlFlowInterfaces.td" // ReturnLike

include "mlir/Interfaces/InferTypeOpInterface.td" // SameOperandsAndResultRank


// Op Base
class FriskGraph_Op<string mnemonic, list<Trait> traits = []> :
  Op<Frisk_Dialect, mnemonic, !listconcat(traits, [Frisk_ExecutionOpInterface])>;

def FriskGraph_MatmulOp : Frisk_Op<"matmul", [Pure]> {
  let summary = "matmul";

  let description = [{
  }];

  let arguments = (ins
    AnyRankedTensor:$lhs,
    AnyRankedTensor:$rhs,
  );

  let results = (outs
    AnyRankedTensor:$result
  );

  let hasVerifier = 1;

  let assemblyFormat =
      "operands attr-dict `:` "
      "functional-type(operands, results)";
}

def FriskGraph_OnlineSoftmaxOp
    : Frisk_Op<"online_softmax", [Pure]> {

  let summary =
      "Incrementally update row-wise online softmax state";

  let description = [{
    Computes an online softmax update over a score tile.

    Inputs:
      scores   : [BM, BN]
      prev_max : [BM]
      prev_sum : [BM]

    Outputs:
      next_max : [BM]
      next_sum : [BM]
      weights  : [BM, BN]
      rescale  : [BM]

    The weights are unnormalized exponential weights.
    The rescale factor updates a separate attention
    accumulator.
  }];

  let arguments = (ins
    AnyRankedTensor:$scores,
    AnyRankedTensor:$prev_max,
    AnyRankedTensor:$prev_sum,
    StrAttr:$exp_base
  );

  let results = (outs
    AnyRankedTensor:$next_max,
    AnyRankedTensor:$next_sum,
    AnyRankedTensor:$weights,
    AnyRankedTensor:$rescale
  );

  let hasVerifier = 1;

  let assemblyFormat =
      "operands attr-dict `:` "
      "functional-type(operands, results)";
}

#endif // FRISK_OPS
