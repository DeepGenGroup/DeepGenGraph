#ifndef DEEPGENGRAPH_FRAGMENT_LDS_PREFETCH_H
#define DEEPGENGRAPH_FRAGMENT_LDS_PREFETCH_H

#include "llvm/ADT/StringRef.h"

namespace mlir::frisk {
// A logical MMA anchor, carried to WarpMmaRR and consumed by ThreadIR. This
// is a compiler scheduling contract, not a workgroup synchronization event.
inline constexpr llvm::StringLiteral FragmentLDSAnchor = "frisk.lds_prefetch_anchor";
inline constexpr llvm::StringLiteral FragmentLDSSchedule = "frisk.fragment_lds_prefetch";
} // namespace mlir::frisk

#endif
