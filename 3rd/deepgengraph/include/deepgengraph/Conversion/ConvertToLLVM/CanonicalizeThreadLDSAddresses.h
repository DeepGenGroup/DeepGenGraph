#ifndef DEEPGENGRAPH_CANONICALIZE_THREAD_LDS_ADDRESSES_H
#define DEEPGENGRAPH_CANONICALIZE_THREAD_LDS_ADDRESSES_H

namespace mlir {
class Operation;
namespace frisk {
// A negative result proves the address pass has no loads to rewrite.
bool hasThreadLDSAddressCandidates(Operation *root);
} // namespace frisk
} // namespace mlir

#endif
