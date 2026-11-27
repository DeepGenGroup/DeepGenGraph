// Shared FP-environment guards for fragment and thread mask-add bypass.
#ifndef DEEPGENGRAPH_CONVERSION_MASKADDBYPASS_H
#define DEEPGENGRAPH_CONVERSION_MASKADDBYPASS_H
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir::frisk::mask_add_bypass {
inline bool isProtected(func::FuncOp function) {
  for (StringRef name : {"strictfp", "llvm.strictfp", "optnone", "llvm.optnone",
                         "optimize_none", "llvm.optimize_none",
                         "denormal-fp-math", "denormal-fp-math-f32",
                         "llvm.denormal_fp_math", "llvm.denormal_fp_math_f32"})
    if (function->hasAttr(name)) return true;
  for (StringRef key : {"passthrough", "llvm.passthrough"})
    if (auto attrs = function->getAttrOfType<ArrayAttr>(key))
      for (Attribute attr : attrs) {
        if (auto pair = dyn_cast<ArrayAttr>(attr))
          if (!pair.empty()) attr = pair[0];
        if (auto name = dyn_cast<StringAttr>(attr))
          if (name.getValue() == "strictfp" || name.getValue() == "optnone" ||
              name.getValue() == "denormal-fp-math" ||
              name.getValue() == "denormal-fp-math-f32") return true;
      }
  return false;
}

} // namespace mlir::frisk::mask_add_bypass
#endif
