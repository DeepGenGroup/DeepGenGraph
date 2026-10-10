#ifndef DEEPGENGRAPH_DIALECT_FRISK_IR_FRISKTRAITS_H
#define DEEPGENGRAPH_DIALECT_FRISK_IR_FRISKTRAITS_H

#include "mlir/IR/OpDefinition.h"

namespace mlir::frisk {

// Results denote computed logical tile values, not aliases of operand storage
// or other existing buffers. A memref result type describes the block tile;
// it does not require materializing the value in that memory space. Lowering
// may keep the value in registers until an explicit store/copy boundary.
// This is independent of memory effects: computing a tile may read operands.
// Views, destination-style updates and copy results must not carry this trait.
template <typename ConcreteType>
class ComputedTile
    : public OpTrait::TraitBase<ConcreteType, ComputedTile> {};

} // namespace mlir::frisk

#endif
