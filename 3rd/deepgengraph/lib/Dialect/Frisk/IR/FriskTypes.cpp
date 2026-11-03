#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "deepgengraph/Dialect/Frisk/IR/FriskDialect.h"
#include "deepgengraph/Dialect/Frisk/IR/FriskTypes.h"

using namespace mlir;
using namespace mlir::frisk;

#define GET_TYPEDEF_CLASSES
#include "deepgengraph/Dialect/Frisk/IR/FriskTypes.cpp.inc"

LogicalResult FragmentType::verify(function_ref<InFlightDiagnostic()> error,
                                   VectorType registers, DictionaryAttr layout) {
  if (registers.isScalable() || registers.getRank() < 1 || registers.getRank() > 2)
    return error() << "fragment requires a fixed rank-one or rank-two register vector";
  if (!layout.empty()) {
    auto role = layout.getAs<StringAttr>("role");
    auto instruction = layout.getAs<StringAttr>("instruction");
    auto widths = layout.getAs<DenseI64ArrayAttr>("thread_widths");
    auto repeat = layout.getAs<DenseI64ArrayAttr>("warp_repeat");
    auto lanes = layout.getAs<DenseI64ArrayAttr>("warp_layout");
    auto wave = layout.getAs<IntegerAttr>("warp_threads");
    if (!role || (role.getValue() != "A" && role.getValue() != "B" && role.getValue() != "C") ||
        !instruction || instruction.getValue().empty() || registers.getRank() != 2 ||
        !widths || widths.size() != 2 || !repeat || repeat.size() != 2 ||
        !lanes || lanes.size() != 2 || !wave || wave.getInt() <= 0)
      return error() << "incomplete MMA fragment layout contract";
    for (unsigned d = 0; d < 2; ++d)
      if (widths[d] <= 0 || repeat[d] <= 0 || lanes[d] <= 0 ||
          registers.getDimSize(d) % widths[d] ||
          registers.getDimSize(d) / widths[d] != repeat[d])
        return error() << "MMA register shape does not match layout";
    if (wave.getInt() % lanes[0] || wave.getInt() / lanes[0] != lanes[1])
      return error() << "MMA lane layout does not match wave size";
    for (StringRef name : {"warp_layout_order", "thread_creg_order", "warp_repeat_order"}) {
      auto order = layout.getAs<DenseI64ArrayAttr>(name);
      if (!order || order.size() != 2 ||
          !((order[0] == 0 && order[1] == 1) || (order[0] == 1 && order[1] == 0)))
        return error() << "invalid MMA layout order " << name;
    }
  }
  return success();
}

void FriskDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "deepgengraph/Dialect/Frisk/IR/FriskTypes.cpp.inc"
      >();
}
