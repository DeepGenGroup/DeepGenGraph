#include "deepgengraph/Analysis/SharedMemoryPlan.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Parser/Parser.h"
#include <cstdlib>

using namespace mlir;
using namespace mlir::frisk;
static void check(bool value, const char *message) {
  if (!value) { llvm::errs() << "FAIL: " << message << '\n'; std::exit(1); }
}

static constexpr const char *fixture = R"mlir(
module {
  func.func @generic(%source: memref<32x64xf16, 1>, %choose: i1)
      attributes {thread_num = 64 : i32} {
    %a = frisk.alloc_buffer {scope = "shared", alignment = 16} -> memref<32x64xf16, 3>
    %b = frisk.alloc_buffer {scope = "shared", alignment = 16} -> memref<32x64xf16, 3>
    %selected = arith.select %choose, %a, %b : memref<32x64xf16, 3>
    %task = frisk.async_copy_gm_to_shm %source to %selected {alignment = 16 : i64} : memref<32x64xf16, 1>, memref<32x64xf16, 3> -> !frisk.async_task
    frisk.wait_buffer_ready %task : !frisk.async_task
    %value = frisk.mul %selected, %selected : memref<32x64xf16, 3>, memref<32x64xf16, 3> -> memref<32x64xf16>
    return
  }
}
)mlir";

static LowerInfoMap makeLayouts(func::FuncOp function) {
  LowerInfoMap layouts;
  function.walk([&](MulOp op) {
    LowerInfo info(64);
    info.base_layout.thread_creg = {1, 4};
    info.base_layout.thread_creg_order = {1, 0};
    info.base_layout.warp_layout = {16, 4};
    info.base_layout.warp_layout_order = {1, 0};
    info.base_layout.warp_repeat = {1, 1};
    info.base_layout.warp_repeat_order = {1, 0};
    info.block_repeat = {2, 4};
    info.thread_own_data_size = {2, 16};
    for (auto &operand : op->getOpOperands()) layouts.setOperandLowerInfo(operand, info);
  });
  return layouts;
}

int main() {
  DialectRegistry registry;
  registry.insert<FriskDialect, func::FuncDialect, arith::ArithDialect,
                  affine::AffineDialect, memref::MemRefDialect, scf::SCFDialect, gpu::GPUDialect>();
  MLIRContext context(registry);
  auto module = parseSourceString<ModuleOp>(fixture, &context);
  check(bool(module), "parse generic elementwise/copy fixture without GEMM or tensor roles");
  auto function = *module->getOps<func::FuncOp>().begin();
  auto layouts = makeLayouts(function);
  auto *hw = GetHWSpecification(HW_KIND_DCU, HW_VERSION_DCU_BW1000, &context);
  auto groups = collectSharedBufferAccesses(function, layouts, *hw);
  check(groups.size() == 1 && groups[0].complete && groups[0].allocations.size() == 2,
        "selected allocations are one complete layout group");
  check(groups[0].accesses.size() == 3, "collect copy and both operands of repeated buffer");
  auto &a = groups[0].accesses[1];
  auto &b = groups[0].accesses[2];
  check(a.op == b.op && a.operand != b.operand, "use identity includes operand number");
  auto *info = layouts.getOperandLowerInfo(a.op->getOpOperand(a.operand));
  auto snapshot = info->getLogicalAccess(64);
  check(snapshot && snapshot->upperBounds == std::array<int64_t,7>{64,2,4,1,1,1,4},
        "logical snapshot includes repeat and register domain");
  auto different = *info;
  different.logicalTransform = AffineMap::get(2, 0,
      {getAffineDimExpr(0, &context), (getAffineDimExpr(1, &context) + 4) % 64}, &context);
  layouts.setOperandLowerInfo(b.op->getOpOperand(b.operand), different);
  check(layouts.getOperandLowerInfo(a.op->getOpOperand(a.operand))->logicalTransform !=
        layouts.getOperandLowerInfo(b.op->getOpOperand(b.operand))->logicalTransform,
        "different access contracts coexist for the same value and op");
  layouts = makeLayouts(function);
  SharedPlanningOptions options;
  options.searchSwizzles = false;
  auto plan = planSharedMemory(function, layouts, *hw, options);
  check(plan.groups.size() == 1 && plan.groups[0].selected, "generic layout/DMA plan selected");
  bool found8 = false;
  for (auto &candidate : plan.groups[0].candidates)
    for (auto &copy : candidate.copies)
      if (copy.bytesPerLane == 8) {
        found8 = true; check(!copy.exportable, "8B capability is distinct from enabled export");
      }
  check(found8, "8B hardware-legal candidates are retained for inspection");
  auto &chosen = plan.groups[0].candidates[*plan.groups[0].selected];
  check(chosen.legal && chosen.exportable && chosen.bankCost && chosen.accesses.size() == 2,
        "each consumer is assessed and unknown is distinct from zero");
  for (auto &copy : chosen.copies) {
    check(copy.copyThreads == 64 && copy.bytesPerLane != 8, "selected export is executable");
    auto src = copy.sourceOffsets;
    for (int64_t p = 0; p < 2048; ++p) {
      auto xy = copy.sourceCoordinates.compose({p});
      check(src.compose({p})[0] == xy[0] * 64 + xy[1], "DMA source follows inverse placement");
    }
  }
  // Analysis is read-only and repeatable.
  for (auto *alloc : groups[0].allocations)
    check(!alloc->hasAttr("frisk.shared_pack"), "analysis does not publish speculative attributes");
  auto again = planSharedMemory(function, layouts, *hw, options);
  check(again.groups[0].selected == plan.groups[0].selected, "deterministic selection");

  // Explicit composed XOR is independent of tensor names and argument order.
  OpBuilder builder(&context);
  for (auto *alloc : groups[0].allocations) {
    alloc->setAttr("frisk.shared_pack", builder.getI64IntegerAttr(8));
    alloc->setAttr("frisk.shared_pack_axis", builder.getI64IntegerAttr(1));
    alloc->setAttr("frisk.shared_swizzles", builder.getDenseI64ArrayAttr({8,6,9,3}));
  }
  auto composed = planSharedMemory(function, layouts, *hw, options);
  check(!composed.groups[0].candidates.empty(), "composed swizzle is representable");
  bool legal8 = false;
  for (auto &candidate : composed.groups[0].candidates)
    for (auto &copy : candidate.copies) legal8 |= candidate.legal && copy.bytesPerLane == 8;
  check(legal8, "composed layout proves direct 8B DMA");
  // Its enabled alternative can use 4B; a forced 8B request must fail atomically.
  AsyncCopyGmToShmOp copy;
  function.walk([&](AsyncCopyGmToShmOp op) { copy = op; });
  copy->setAttr("frisk.dma_bytes", builder.getI64IntegerAttr(8));
  auto forced = planSharedMemory(function, layouts, *hw, options);
  check(!forced.groups[0].selected, "unsupported forced exporter is not silently narrowed");
  check(failed(forced.publish(layouts)), "unexportable explicit request rejects publication");
  for (auto *alloc : groups[0].allocations) check(!alloc->hasAttr("frisk.shared_plan"), "failed plan publishes nothing");

  auto bufferExporter = options;
  bufferExporter.exportableDMABytes = {4, 8, 16};
  auto enabled8 = planSharedMemory(function, layouts, *hw, bufferExporter);
  check(enabled8.groups[0].selected.has_value(), "buffer exporter accepts explicit composed 8B DMA");
  auto &dma8 = enabled8.groups[0].candidates[*enabled8.groups[0].selected].copies[0];
  check(dma8.bytesPerLane == 8 && succeeded(enabled8.publish(layouts)), "publish 8B copy plan");
  check(copy->getAttrOfType<AffineMapAttr>("frisk.dma_destination_map").getValue() == dma8.destination &&
        copy->getAttrOfType<AffineMapAttr>("frisk.dma_source_map").getValue() == dma8.sourceCoordinates,
        "IR emission receives the exact analyzed ownership and source maps");

  // A new unknown access invalidates the entire alias group, including direct users.
  auto clean = parseSourceString<ModuleOp>(fixture, &context);
  auto fn = *clean->getOps<func::FuncOp>().begin();
  auto cleanLayouts = makeLayouts(fn);
  auto valid = planSharedMemory(fn, cleanLayouts, *hw, options);
  builder.setInsertionPoint(fn.front().getTerminator());
  builder.create<memref::DimOp>(fn.getLoc(), valid.groups[0].group.aliases.front(), 0);
  check(failed(valid.publish(cleanLayouts)), "new access invalidates snapshot before any publication");
  auto unknown = planSharedMemory(fn, cleanLayouts, *hw, options);
  check(!unknown.groups[0].group.complete && !unknown.groups[0].selected,
        "unknown access cannot be omitted from legality");

  auto publishModule = parseSourceString<ModuleOp>(fixture, &context);
  auto publishFn = *publishModule->getOps<func::FuncOp>().begin();
  auto publishLayouts = makeLayouts(publishFn);
  auto publishPlan = planSharedMemory(publishFn, publishLayouts, *hw, options);
  check(succeeded(publishPlan.publish(publishLayouts)), "publish all proven aliases and copy plans");
  for (auto *alloc : publishPlan.groups[0].group.allocations)
    check(alloc->hasAttr("frisk.shared_plan"), "every alternative receives the final layout");
  auto scalar = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @scalar() attributes {thread_num = 64 : i32} {
        %data = memref.alloc() alignment = 16 : memref<2x64xf32, 3>
        %tid = gpu.thread_id x
        %zero = arith.constant 0.0 : f32
        affine.for %row = 0 to 2 {
          affine.store %zero, %data[%row, %tid] : memref<2x64xf32, 3>
          %loaded = affine.load %data[%row, %tid] : memref<2x64xf32, 3>
        }
        return
      }
    }
  )mlir", &context);
  check(bool(scalar), "parse explicit scalar shared-memory accesses");
  LowerInfoMap noMmaLayouts;
  auto scalarPlan = planSharedMemory(*scalar->getOps<func::FuncOp>().begin(), noMmaLayouts, *hw, options);
  check(scalarPlan.groups.size() == 1 && scalarPlan.groups[0].group.complete && scalarPlan.groups[0].selected,
        "affine reads and writes are planned without GEMM or LowerInfo anchors");
  auto &scalarChoice = scalarPlan.groups[0].candidates[*scalarPlan.groups[0].selected];
  check(scalarChoice.accesses.size() == 2 && scalarChoice.bankCost,
        "both scalar producer and consumer participate in bank evaluation");

  // An outer loop round may end in inactive waves, but never inactive lanes
  // inside a selected DMA wave. Inspect the actual destination coverage.
  std::string tailText = fixture;
  for (size_t at = 0; (at = tailText.find("32x64", at)) != std::string::npos; at += 5)
    tailText.replace(at, 5, "24x32");
  tailText.replace(tailText.find("thread_num = 64"), std::string("thread_num = 64").size(), "thread_num = 128");
  auto tailModule = parseSourceString<ModuleOp>(tailText, &context);
  check(bool(tailModule), "parse whole-wave tail fixture");
  auto tailFn = *tailModule->getOps<func::FuncOp>().begin();
  tailFn.walk([&](AsyncCopyGmToShmOp op) { op->setAttr("frisk.dma_bytes", builder.getI64IntegerAttr(8)); });
  auto tailLayouts = makeLayouts(tailFn);
  auto bufferExportWithTail = options;
  bufferExportWithTail.exportableDMABytes = {4, 8, 16}; // Actual buffer-asm capability.
  auto tailPlan = planSharedMemory(tailFn, tailLayouts, *hw, bufferExportWithTail);
  if (!tailPlan.groups[0].selected) tailPlan.dump(llvm::errs());
  check(tailPlan.groups[0].selected.has_value(), "tail can use enabled 8B DMA");
  auto &tailCopy = tailPlan.groups[0].candidates[*tailPlan.groups[0].selected].copies[0];
  check(tailCopy.rounds == 2 && tailCopy.copyThreads == 128, "round count includes partial final workgroup");
  std::vector<unsigned> covered(tailCopy.totalElements, 0);
  for (int64_t round = 0; round < int64_t(tailCopy.rounds); ++round)
    for (int64_t tid = 0; tid < tailCopy.copyThreads; ++tid) {
      auto p = tailCopy.destination.compose({tid, round})[0];
      if (uint64_t(p) >= tailCopy.totalElements) continue;
      for (unsigned e = 0; e < 4; ++e) ++covered[p + e];
    }
  check(llvm::all_of(covered, [](unsigned n) { return n == 1; }), "tail DMA covers every element exactly once");

  llvm::outs() << "PASS generic shared access collection, joint plans, capabilities and atomic publication\n";
}
