#ifndef DEEPGENGRAPH_THREAD_SCHEDULING_H
#define DEEPGENGRAPH_THREAD_SCHEDULING_H
#include <memory>
#include <string>

namespace mlir {
class Pass;
namespace frisk {
struct MMAChainInterleaveOptions {
  unsigned width = 0;
  int guardGroup = 2;
  std::string operandSource = "any"; // any, register, lds, matrix
  std::string loadMotion = "none";   // none, bounded (ordinary LDS only)
  unsigned prefetchWords = 32;       // Distinct B-packet footprint, words/thread.
};
struct PairedLDSPrefetchOptions {
  unsigned distance = 0;             // Maximum number of intervening MMAs.
  std::string window = "register";   // address, register, reduction; fragment finalizes an earlier plan
};
std::unique_ptr<Pass> createThreadInterleaveMMAChainsPass(
    const MMAChainInterleaveOptions &options = {});
std::unique_ptr<Pass> createThreadPrefetchPairedLDSPass(
    const PairedLDSPrefetchOptions &options = {});
void registerLegacyThreadSchedulingPipelines();
} // namespace frisk
} // namespace mlir
#endif
