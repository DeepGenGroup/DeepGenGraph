#include "deepgengraph/Analysis/LowerInfo.h"
#include "mlir/IR/Block.h"
#include <cstdlib>
#include <set>

using namespace mlir;
using namespace mlir::frisk;

static void check(bool ok, const char *message) {
  if (!ok) {
    llvm::errs() << "FAIL: " << message << '\n';
    std::exit(1);
  }
}

static void checkBijection(const ShmDataIODesc &io, MLIRContext &ctx) {
  auto forward = io.getLogicalToPhysicalMap(&ctx);
  auto inverse = io.getPhysicalToLogicalMap(&ctx);
  auto global = io.getGlobalSourceMap(&ctx, {193, 2}, 17);
  std::set<int64_t> offsets;
  for (int64_t row = 0; row < io.shape[0]; ++row)
    for (int64_t col = 0; col < io.shape[1]; ++col) {
      int64_t p = forward.compose({row, col})[0];
      check(p >= 0 && p < io.shape[0] * io.shape[1], "physical bounds");
      check(offsets.insert(p).second, "no duplicate physical elements");
      auto xy = inverse.compose({p});
      check(xy[0] == row && xy[1] == col, "forward/inverse agreement");
      check(global.compose({p})[0] == 17 + 193 * row + 2 * col,
            "global source offset uses actual strides and view origin");
    }
}

int main() {
  MLIRContext ctx;
  auto *nv = GetHWSpecification(HW_KIND_NVIDIA, HW_VERSION_NV_A100, &ctx);
  auto *hw = GetHWSpecification(HW_KIND_DCU, HW_VERSION_DCU_BW1000, &ctx);
  check(hw != nv && hw->getWarpsize() == 64 && nv->getWarpsize() == 32,
        "hardware queries are keyed by target");
  check(hw == GetHWSpecification(HW_KIND_DCU, HW_VERSION_DCU_BW1000, &ctx),
        "hardware descriptor lifetime is stable");
  check(hw->dataCopyInfo.bankcount == 32 && hw->dataCopyInfo.supportAsyncCopy,
        "bank count is an integer and DMA is supported");
  check(hw->dataCopyInfo.getDMATransfer(16) &&
        hw->dataCopyInfo.getDMATransfer(16)->ldsLaneStrideBytes == 16 &&
        !hw->dataCopyInfo.getDMATransfer(8), "explicit DMA width set");
  for (auto kind : {ShmAccessKind::Read, ShmAccessKind::Write})
    for (unsigned bytes : {4, 8, 16}) {
      auto phases = hw->getBankConflictPhases(bytes, kind);
      check(phases && phases->size() == 64 / (128 / bytes), "phase count");
      std::set<unsigned> lanes;
      for (const auto &phase : *phases) {
        check(phase.size() == 128 / bytes, "phase lane count");
        for (unsigned lane : phase)
          check(lane < 64 && lanes.insert(lane).second, "phase lane partition");
      }
      check(lanes.size() == 64, "all wave lanes covered");
    }
  auto read128 = hw->getBankConflictPhases(16, ShmAccessKind::Read);
  auto write128 = hw->getBankConflictPhases(16, ShmAccessKind::Write);
  check((*read128)[0] == std::vector<unsigned>({0,1,2,3,20,21,22,23}) &&
        (*write128)[0] == std::vector<unsigned>({0,1,2,3,4,5,6,7}),
        "b128 reads and writes use different lane groups");
  check(!hw->getBankConflictPhases(12, ShmAccessKind::Read) &&
        !nv->getBankConflictPhases(8, ShmAccessKind::Read),
        "unknown rules are not treated as conflict-free");

  for (int64_t element : {1, 2, 4})
    for (int64_t axis : {0, 1})
      for (coordXY_t shape : {coordXY_t{32, 64}, coordXY_t{48, 128}}) {
        auto sw = *ShmDataIODesc::getPacked(shape, element, axis, 16 / element);
        check(sw.setXorSwizzle(8, 6), "legal packet swizzle");
        checkBijection(sw, ctx);
        for (int64_t p = 0; p < shape[0] * shape[1]; ++p) {
          auto e = sw.swizzleElementOffset(getAffineDimExpr(0, &ctx));
          auto map = AffineMap::get(1, 0, e);
          int64_t want = (p * element) ^ (((p * element) >> 2) & 64);
          check(map.compose({p})[0] * element == want, "byte-bit XOR semantics");
          check(map.compose({p})[0] % (16 / element) == p % (16 / element),
                "all intra-DMA packet elements retain their offsets");
        }
        check(!sw.setXorSwizzle(7, 3), "reject Q subpacket permutation");
        check(!sw.setXorSwizzle(6, 6), "reject nonbijective equal bits");
        check(!sw.setXorSwizzle(31, 6), "reject oversized periods");
      }
  auto tail = *ShmDataIODesc::getPacked({12, 8}, 2, 1, 8);
  check(!tail.setXorSwizzle(8, 6), "reject incomplete permutation period");
  auto configurable = *ShmDataIODesc::getPacked({32, 64}, 2, 0, 8);
  for (int source = 4; source <= 9; ++source)
    for (int target = 4; target <= 9; ++target) {
      if (source == target) continue;
      check(configurable.setXorSwizzle(source, target), "configurable byte bits");
      checkBijection(configurable, ctx);
    }

  auto q = *ShmDataIODesc::getPacked({128,32}, 2, 1, 4);
  auto wide = *ShmDataIODesc::getPacked({128,32}, 2, 1, 8);
  checkBijection(q, ctx);
  checkBijection(wide, ctx);
  for (int64_t bytes : {2, 4}) {
    auto packets = *ShmDataIODesc::getPacked({128,64}, bytes, 1, 16 / bytes, 0, 0, 0, 2);
    checkBijection(packets, ctx);
    check(packets.selectDMATransferBytes(*hw, &ctx, {128,1}, 16) == 16,
          "interleaved row packets retain direct 16B DMA");
  }
  check(!ShmDataIODesc::getPacked({128,64}, 2, 0, 8, 0, 0, 0, 2) &&
        !ShmDataIODesc::getPacked({128,64}, 2, 1, 8, 0, 0, 0, 3),
        "invalid packet row groups rejected");
  checkBijection(*ShmDataIODesc::getPacked({32,128}, 2, 0, 4), ctx);
  checkBijection(*ShmDataIODesc::getPacked({32,128}, 2, 0, 4, 16), ctx);
  checkBijection(*ShmDataIODesc::getPacked({128,32}, 2, 0, 4, 4), ctx);
  checkBijection(*ShmDataIODesc::getPacked({16,32}, 2, 1, 4, 0, 4, 16), ctx);
  checkBijection(*ShmDataIODesc::getPacked({16,10}, 4, 0, 2, 4), ctx);
  check(!ShmDataIODesc::getPacked({16,31}, 2, 1, 4) &&
        !ShmDataIODesc::getPacked({16,32}, 2, 0, 4, 6) &&
        !ShmDataIODesc::getPacked({16,32}, 2, 1, 4, 0, 3, 16),
        "invalid packing rejected");
  check(q.selectDMATransferBytes(*hw, &ctx, {128,1}, 16) == 4,
        "Q four-element packets cross GM rows at 16B width");
  check(wide.selectDMATransferBytes(*hw, &ctx, {128,1}, 16) == 16,
        "eight-element Q packets allow 16B DMA");
  check(wide.selectDMATransferBytes(*hw, &ctx, {128,1}, 4) == 4,
        "source alignment limits DMA width");
  check(wide.selectDMATransferBytes(*hw, &ctx, {128,2}, 16) == 0 &&
        wide.selectDMATransferBytes(*hw, &ctx, {128,1}, 2) == 0,
        "noncontiguous and underaligned sources reject DMA");
  auto b = *ShmDataIODesc::getPacked({32,128}, 2, 0, 8);
  check(b.selectDMATransferBytes(*hw, &ctx, {1,128}, 16) == 16,
        "transposed source supports physical-minor-axis DMA");

  std::vector<int64_t> addresses(64);
  for (int64_t rows : {0, 2}) {
    auto io = *ShmDataIODesc::getPacked({128,64}, 2, 1, 8, 0, 0, 0, rows);
    auto physical = io.getLogicalToPhysicalMap(&ctx);
    for (int64_t r = 0; r < 128; r += 16)
      for (int64_t c = 0; c < 64; c += 32) {
        for (int64_t lane = 0; lane < 64; ++lane) {
          int64_t row = r + lane / 4;
          row = row / 64 * 64 + (row / 4 % 4) * 16 + (row / 16 % 4) * 4 + row % 4;
          addresses[lane] = physical.compose({row, c + lane % 4 * 8})[0] * 2;
        }
        for (auto kind : {ShmAccessKind::Read, ShmAccessKind::Write})
          check(io.getBankConflictWays(*hw, addresses, kind) == (rows ? 1 : 4),
                "matrix lane packet address model improves four-way to one-way");
      }
  }
  for (int64_t lane = 0; lane < 64; ++lane) addresses[lane] = 8 * lane;
  check(q.getBankConflictWays(*hw, addresses) == 1, "contiguous b64 reads");
  for (int64_t lane = 0; lane < 64; ++lane) addresses[lane] = 16 * lane;
  check(q.getBankConflictWays(*hw, addresses) == 2,
        "wide packing can introduce two-way b64 read conflicts");
  check(wide.getBankConflictWays(*hw, addresses, ShmAccessKind::Write) == 1,
        "same wide packing has conflict-free b128 writes");
  addresses.assign(64, 0);
  check(q.getBankConflictWays(*hw, addresses) == 1 &&
        q.getBankConflictWays(*hw, addresses, ShmAccessKind::Write) == 16,
        "read broadcast does not apply to overlapping writes");
  addresses.assign(64, -1);
  check(q.getBankConflictWays(*hw, addresses) == 0, "inactive lanes");
  addresses[0] = 1;
  check(!q.getBankConflictWays(*hw, addresses), "unaligned access is unknown");

  Block block;
  LowerInfo info(64);
  info.buffer = block.addArgument(MemRefType::get({16,32}, Float16Type::get(&ctx)),
                                 UnknownLoc::get(&ctx));
  info.base_layout = hw->gemmInfo.validInsts[1].desc_a;
  info.block_repeat = {1,2};
  info.thread_own_data_size = {1,8};
  info.shmIO = ShmDataIODesc::getPacked({16,32}, 2, 1, 4);
  info.shmIO->baseLayout = info.base_layout;
  auto map = info.getShmAccessMap();
  for (int64_t lane = 0; lane < 64; ++lane)
    for (int64_t kfrag = 0; kfrag < 2; ++kfrag)
      for (int64_t reg = 0; reg < 4; ++reg) {
        auto logical = info.getAffineMap().compose({lane,0,kfrag,0,0,0,reg});
        auto physical = map.compose({lane,0,kfrag,0,0,0,reg})[0];
        check(physical == ((logical[1]/4)*16 + logical[0])*4 + logical[1]%4,
              "LowerInfo consumer map composes all iteration coordinates");
      }
  LowerInfo copy = info;
  info.base_layout.thread_creg = {4,1};
  check(copy.shmIO->baseLayout->thread_creg == coordXY_t{1,4},
        "descriptor owns its consumer snapshot");
  check(info.convert_to_row_major() && !info.shmIO,
        "transpose invalidates physical plan");
  llvm::outs() << "PASS hardware capabilities, phases, LDS maps, DMA planning and conflicts\n";
}
