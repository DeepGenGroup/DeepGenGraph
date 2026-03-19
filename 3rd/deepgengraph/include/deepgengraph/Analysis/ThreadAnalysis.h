#ifndef THREAD_ANALYSIS_H
#define THREAD_ANALYSIS_H

#include "deepgengraph/Dialect/DeepgengraphTriton/IR/DeepgengraphTritonDialect.h"

namespace analyze {

struct PtrInfo {
  mlir::deepgengraph::triton::BlockPointerOfOp m_blockPtrOfOp;
  mlir::deepgengraph::triton::PointerOfOp m_pointerOfOp;
};

using PtrInfoMapType = std::map<mlir::deepgengraph::triton::BlockLoadOp, PtrInfo> ;
class PointerTracer {
private:
  static PtrInfoMapType m_map;
  static mlir::deepgengraph::triton::BlockPointerOfOp findBlockPointerOfBlockLoadOp(mlir::deepgengraph::triton::BlockLoadOp op);
  static mlir::deepgengraph::triton::PointerOfOp findPointerOfOp(mlir::deepgengraph::triton::BlockPointerOfOp op);
public:
  static void getPointerInfo(mlir::ModuleOp mod);
  static const PtrInfoMapType& getMap();
};

}

#endif