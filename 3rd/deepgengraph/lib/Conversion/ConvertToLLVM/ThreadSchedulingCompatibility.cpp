#include "deepgengraph/Conversion/ConvertToLLVM/Passes.h"
#include "deepgengraph/Conversion/ConvertToLLVM/PVInterleaveOptions.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir::frisk {
namespace {
template<bool Reads>
struct LegacyMMAOptions : PassPipelineOptions<LegacyMMAOptions<Reads>> {
  using Base = PassPipelineOptions<LegacyMMAOptions<Reads>>;
  typename Base::template Option<unsigned> width{*this, "width", llvm::cl::init(0)};
  typename Base::template Option<int> guardGroup{*this, "guard-group", llvm::cl::init(Reads ? 2 : -1)};
  typename Base::template Option<unsigned> prefetchWords{*this, "prefetch-words", llvm::cl::init(32)};
};
struct LegacyLDSOptions : PassPipelineOptions<LegacyLDSOptions> {
  Option<unsigned> distance{*this, "distance", llvm::cl::init(0)};
};
} // namespace
std::unique_ptr<Pass> createThreadInterleavePVPass(unsigned width, int guards) {
  return createThreadInterleaveMMAChainsPass({width, guards, "matrix", "none", 32});
}
std::unique_ptr<Pass> createThreadInterleaveQKPass(unsigned width, int guards, unsigned words) {
  return createThreadInterleaveMMAChainsPass({width, guards, "lds", "bounded", words});
}
std::unique_ptr<Pass> createThreadPrefetchPairedLDSPass(unsigned distance) {
  return createThreadPrefetchPairedLDSPass(PairedLDSPrefetchOptions{distance, "address"});
}
std::unique_ptr<Pass> createThreadPrefetchPairedLDSTailPass(unsigned distance) {
  return createThreadPrefetchPairedLDSPass(PairedLDSPrefetchOptions{distance, "reduction"});
}
void registerLegacyThreadSchedulingPipelines() {
  static PassPipelineRegistration<LegacyMMAOptions<false>> pv(
      "frisk-thread-interleave-pv", "Compatibility alias for matrix-fed MMA interleaving",
      [](OpPassManager &pm, const LegacyMMAOptions<false> &o) {
        pm.addPass(createThreadInterleavePVPass(o.width, o.guardGroup));
      });
  static PassPipelineRegistration<LegacyMMAOptions<true>> qk(
      "frisk-thread-interleave-qk", "Compatibility alias for bounded LDS-fed MMA interleaving",
      [](OpPassManager &pm, const LegacyMMAOptions<true> &o) {
        pm.addPass(createThreadInterleaveQKPass(o.width, o.guardGroup, o.prefetchWords));
      });
  static PassPipelineRegistration<LegacyLDSOptions> tail(
      "frisk-thread-prefetch-paired-lds-tail", "Compatibility alias for reduction-window LDS pairing",
      [](OpPassManager &pm, const LegacyLDSOptions &o) {
        pm.addPass(createThreadPrefetchPairedLDSTailPass(o.distance));
      });
  (void)pv; (void)qk; (void)tail;
}
bool parsePVInterleaveOption(llvm::StringRef argument, unsigned &width, bool &consumed) {
  consumed = argument.starts_with("--pv-interleave");
  if (!consumed) return true;
  unsigned parsed;
  if (!argument.consume_front("--pv-interleave=") || argument.getAsInteger(10, parsed) ||
      (parsed != 0 && parsed != 1 && parsed != 2 && parsed != 4)) return false;
  width = parsed;
  return true;
}

bool parseMMAChainScheduleOption(llvm::StringRef argument,
                                MMAChainScheduleOptions &options, bool &consumed) {
  consumed = true;
  if (argument.starts_with("--qk-interleave")) {
    unsigned value;
    if (!argument.consume_front("--qk-interleave=") || argument.getAsInteger(10, value) ||
        (value != 0 && value != 1 && value != 2 && value != 4)) return false;
    options.qkWidth = value;
  } else if (argument.starts_with("--qk-prefetch-words")) {
    unsigned value;
    if (!argument.consume_front("--qk-prefetch-words=") || argument.getAsInteger(10, value))
      return false;
    options.qkPrefetchWords = value;
  } else if (argument.starts_with("--pv-guard-group") || argument.starts_with("--qk-guard-group")) {
    bool qk = argument.starts_with("--qk-");
    int value;
    if (!argument.consume_front(qk ? "--qk-guard-group=" : "--pv-guard-group=") ||
        argument.getAsInteger(10, value) || value < -1 || value > 32) return false;
    (qk ? options.qkGuardGroup : options.pvGuardGroup) = value;
  } else consumed = false;
  return true;
}

} // namespace mlir::frisk
