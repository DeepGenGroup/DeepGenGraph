#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <vector>

namespace mlir::frisk {

void dumpRegPressure(
    func::FuncOp funcOp,
    unsigned threadNum = 64,
    bool printFullOp = false) ;

} // namespace mlir:frisk
