#include "deepgengraph/Dialect/Frisk/Utils/Utils.h"

namespace mlir::frisk {

int WgmmaConfig::mma_m = 64;  // 64 elements
int WgmmaConfig::mma_k_bytes = 32;  // 32 Bytes

}