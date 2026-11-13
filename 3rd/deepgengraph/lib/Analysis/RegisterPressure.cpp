#include "deepgengraph/Analysis/RegisterPressure.h"
#include "deepgengraph/Dialect/Frisk/IR/FriskDialect.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>

namespace mlir::frisk {
namespace {
using Values = llvm::SmallSetVector<Value, 32>;

static void add(RegisterPressureUsage &a, const RegisterPressureUsage &b) {
  a.dataWords += b.dataWords;
  a.packedDataWords += b.packedDataWords;
  a.descriptorWords += b.descriptorWords;
  a.unknownValues += b.unknownValues;
  a.dataBits += b.dataBits;
  a.addressWords += b.addressWords;
}

static uint64_t words(uint64_t bits) { return (bits + 31) / 32; }

static bool isDescriptor(LLVM::LLVMStructType type) {
  if (type.isOpaque()) return false;
  auto fields = type.getBody();
  if (fields.size() != 5 || !isa<LLVM::LLVMPointerType>(fields[0]) ||
      fields[0] != fields[1] || !isa<IntegerType>(fields[2])) return false;
  auto sizes = dyn_cast<LLVM::LLVMArrayType>(fields[3]);
  auto strides = dyn_cast<LLVM::LLVMArrayType>(fields[4]);
  return sizes && strides && sizes == strides && sizes.getElementType() == fields[2];
}

static RegisterPressureUsage typeCost(Type type, const DataLayout &layout) {
  if (auto fragment = dyn_cast<FragmentType>(type))
    type = fragment.getRegisterType();
  if (isa<friskAsyncTaskType>(type)) return {}; // compile-time dependency only
  uint64_t elements = 1;
  if (auto vector = dyn_cast<VectorType>(type)) {
    if (vector.isScalable()) return {0, 0, 0, 1};
    elements = vector.getNumElements();
    type = vector.getElementType();
  }
  if (type.isIntOrIndexOrFloat()) {
    uint64_t bits = layout.getTypeSizeInBits(type).getFixedValue();
    return {elements * words(bits), words(elements * bits), 0, 0, elements * bits};
  }
  if (auto pointer = dyn_cast<LLVM::LLVMPointerType>(type))
    return {0, 0, 0, 0, 0, words(layout.getTypeSizeInBits(pointer).getFixedValue())};
  if (auto array = dyn_cast<LLVM::LLVMArrayType>(type)) {
    auto usage = typeCost(array.getElementType(), layout);
    auto n = array.getNumElements();
    return {n * usage.dataWords, n * usage.packedDataWords,
            n * usage.descriptorWords, n ? usage.unknownValues : 0,
            n * usage.dataBits, n * usage.addressWords};
  }
  if (auto structure = dyn_cast<LLVM::LLVMStructType>(type)) {
    if (structure.isOpaque()) return {0, 0, 0, 1};
    RegisterPressureUsage usage;
    for (Type field : structure.getBody()) add(usage, typeCost(field, layout));
    if (isDescriptor(structure)) {
      usage.descriptorWords += usage.dataWords + usage.addressWords;
      usage.addressWords = 0;
      usage.dataWords = usage.packedDataWords = usage.dataBits = 0;
    }
    // Unknown components make this one aggregate value unmodeled.
    usage.unknownValues = usage.unknownValues != 0;
    return usage;
  }
  if (auto memref = dyn_cast<MemRefType>(type)) {
    unsigned space = 0;
    if (auto attr = dyn_cast_or_null<IntegerAttr>(memref.getMemorySpace()))
      space = attr.getInt();
    else if (memref.getMemorySpace()) return {0, 0, 0, 1};
    // Standard ranked descriptor: allocated ptr, aligned ptr, offset, sizes,
    // strides. Static components and shared view descriptors may fold later;
    // keep this diagnostic separate from the register pressure estimate.
    auto ptr = LLVM::LLVMPointerType::get(type.getContext(), space);
    auto index = IndexType::get(type.getContext());
    uint64_t pointerWords = words(layout.getTypeSizeInBits(ptr).getFixedValue());
    uint64_t indexWords = words(layout.getTypeSizeInBits(index).getFixedValue());
    return {0, 0, 2 * pointerWords + (1 + 2 * memref.getRank()) * indexWords, 0};
  }
  return {0, 0, 0, 1}; // Never silently count an unsupported payload as zero.
}

static RegisterPressureUsage cost(Value value, const DataLayout &layout) {
  // Constants can be rematerialized. No claim is made about the temporary
  // registers needed to materialize them in the eventual machine program.
  if (matchPattern(value, m_Constant())) return {};
  return typeCost(value.getType(), layout);
}

static Type registerType(Type type) {
  if (auto fragment = dyn_cast<FragmentType>(type)) return fragment.getRegisterType();
  return type;
}

static uint64_t elementCount(Type type) {
  if (auto vector = dyn_cast<VectorType>(registerType(type)))
    return vector.isScalable() ? 0 : vector.getNumElements();
  return 1;
}

static void appendSlice(RegisterPressureStorage &to, RegisterPressureSlice slice) {
  if (!to.slices.empty()) {
    auto &last = to.slices.back();
    if (slice.bitWidth && last.origin == slice.origin &&
        last.offset + last.bitWidth == slice.offset) {
      last.bitWidth += slice.bitWidth;
      return;
    }
  }
  to.slices.push_back(slice);
}

// Project a logical bit range through an exact representation mapping.
static void appendRange(RegisterPressureStorage &to,
                        const RegisterPressureStorage &from,
                        uint64_t offset, uint64_t width) {
  assert(from.exact);
  for (auto slice : from.slices) {
    if (offset >= slice.bitWidth) { offset -= slice.bitWidth; continue; }
    uint64_t take = std::min(width, slice.bitWidth - offset);
    appendSlice(to, {slice.origin, slice.offset + offset, take});
    width -= take;
    if (!width) return;
    offset = 0;
  }
  assert(!width && "range outside representation");
}

// Return source element indices in row-major slice order. Fragment coordinates
// are converted to element offsets by the caller; vector.extract uses a prefix.
static std::optional<SmallVector<uint64_t>> sliceIndices(
    Type sourceType, Type sliceType, ArrayRef<int64_t> offsets) {
  auto source = dyn_cast<VectorType>(registerType(sourceType));
  if (!source || source.isScalable() || offsets.size() > source.getRank()) return {};
  SmallVector<int64_t> shape(source.getRank(), 1), start(offsets);
  start.resize(source.getRank(), 0);
  if (auto slice = dyn_cast<VectorType>(registerType(sliceType))) {
    if (slice.isScalable() || slice.getRank() > source.getRank()) return {};
    llvm::copy(slice.getShape(), shape.end() - slice.getRank());
  }
  for (unsigned d = 0; d < shape.size(); ++d)
    if (start[d] < 0 || start[d] > source.getDimSize(d) - shape[d]) return {};
  SmallVector<uint64_t> indices;
  for (uint64_t i = 0, n = elementCount(sliceType); i < n; ++i) {
    uint64_t rest = i, index = 0, stride = 1;
    for (int d = shape.size() - 1; d >= 0; --d) {
      index += (start[d] + rest % shape[d]) * stride;
      rest /= shape[d];
      stride *= source.getDimSize(d);
    }
    indices.push_back(index);
  }
  return indices;
}

struct Analyzer {
  RegisterPressureResult result;
  const Liveness &liveness;
  DenseMap<Value, unsigned> valueOrder;

  Analyzer(func::FuncOp function, const Liveness &liveness) : liveness(liveness) {
    result.function = function;
    function->walk<WalkOrder::PreOrder>([&](Block *block) {
      for (Value arg : block->getArguments()) valueOrder.try_emplace(arg, valueOrder.size());
      for (Operation &op : *block)
        for (Value value : op.getResults()) valueOrder.try_emplace(value, valueOrder.size());
    });
    // Keep costs scope-sensitive: DLTI may override index/pointer sizes.
    for (auto [value, id] : valueOrder) {
      Operation *scope = value.getDefiningOp();
      if (!scope) scope = cast<BlockArgument>(value).getOwner()->getParentOp();
      auto layout = DataLayout::closest(scope);
      result.costs[value] = cost(value, layout);
      Type type = registerType(value.getType());
      if (auto vector = dyn_cast<VectorType>(type))
        if (!vector.isScalable()) type = vector.getElementType();
      if (type.isIntOrIndexOrFloat())
        result.elementBits[value] = layout.getTypeSizeInBits(type).getFixedValue();
    }
    for (auto [value, id] : valueOrder) { root(value); storageFor(value); }
  }

  RegisterPressureStorage storageFor(Value value) {
    auto found = result.storage.find(value);
    if (found != result.storage.end()) return found->second;
    uint64_t bits = result.elementBits.lookup(value);
    RegisterPressureStorage payload;
    payload.slices.push_back({value, 0, bits * elementCount(value.getType())});
    payload.exact = bits != 0;
    Operation *op = value.getDefiningOp();
    if (bits && op) {
      if (isa<PackFragOp, UnpackFragOp, vector::ShapeCastOp, vector::BitCastOp>(op)) {
        payload = storageFor(op->getOperand(0));
      } else if (isa<vector::FromElementsOp>(op)) {
        payload.slices.clear();
        for (Value operand : op->getOperands()) {
          auto part = storageFor(operand);
          for (auto slice : part.slices) appendSlice(payload, slice);
          payload.exact &= part.exact;
        }
      } else if (auto broadcast = dyn_cast<vector::BroadcastOp>(op)) {
        // Scalar splats have a single content identity, even when represented
        // in many vector positions. Instruction-constrained copies come later.
        if (!isa<VectorType>(broadcast.getSource().getType())) {
          auto part = storageFor(broadcast.getSource());
          payload.slices.clear();
          payload.exact = part.exact;
          for (uint64_t i = 0; i < elementCount(value.getType()); ++i)
            for (auto slice : part.slices) appendSlice(payload, slice);
        }
      } else {
        Value source, destination;
        std::optional<SmallVector<int64_t>> offsets;
        if (auto extract = dyn_cast<vector::ExtractOp>(op)) {
          source = extract.getSource();
          offsets = getConstantIntValues(extract.getMixedPosition());
        } else if (auto insert = dyn_cast<vector::InsertOp>(op)) {
          source = insert.getValueToStore();
          destination = insert.getDest();
          offsets = getConstantIntValues(insert.getMixedPosition());
        } else if (isa<ExtractFragOp, InsertFragOp>(op)) {
          ValueRange indices;
          VectorType fragment;
          if (auto extract = dyn_cast<ExtractFragOp>(op)) {
            source = extract.getSource();
            indices = extract.getIndices();
            fragment = cast<FragmentType>(value.getType()).getRegisterType();
          } else {
            auto insert = cast<InsertFragOp>(op);
            source = insert.getValue();
            destination = insert.getDestination();
            indices = insert.getIndices();
            fragment = cast<FragmentType>(source.getType()).getRegisterType();
          }
          SmallVector<OpFoldResult> positions;
          for (Value index : indices) positions.push_back(index);
          offsets = getConstantIntValues(positions);
          if (offsets) {
            auto tile = cast<VectorType>(destination ? destination.getType() : source.getType());
            for (auto [d, offset] : llvm::enumerate(*offsets)) {
              if (offset < 0 || offset > (tile.getDimSize(d) - fragment.getDimSize(d)) / fragment.getDimSize(d)) {
                offsets.reset();
                break;
              }
              offset *= fragment.getDimSize(d);
            }
          }
        }
        if (source) {
          auto input = storageFor(source);
          auto mapping = offsets ? sliceIndices(
              destination ? destination.getType() : source.getType(),
              destination ? source.getType() : value.getType(), *offsets) : std::nullopt;
          if (!destination) {
            if (mapping && input.exact) {
              payload.slices.clear();
              for (uint64_t index : *mapping) appendRange(payload, input, index * bits, bits);
            } else {
              // A dynamic selector denotes a subset, not another full payload.
              // When its source is absent, charge an independent slice because
              // overlap with other unresolved selectors cannot be proved.
              result.subsetSources[value] = source;
              ++result.unresolvedExtracts;
            }
          } else {
            auto dest = storageFor(destination);
            if (mapping && input.exact && dest.exact) {
              payload.slices.clear();
              unsigned replacement = 0;
              for (uint64_t i = 0; i < elementCount(value.getType()); ++i) {
                if (replacement < mapping->size() && (*mapping)[replacement] == i)
                  appendRange(payload, input, replacement++ * bits, bits);
                else appendRange(payload, dest, i * bits, bits);
              }
            } else {
              // Unknown position: every output bit comes from dest or input.
              // Retaining their union overcounts only potentially overwritten
              // content, rather than charging an entirely new destination.
              payload = std::move(dest);
              for (auto slice : input.slices) appendSlice(payload, slice);
              payload.exact = false;
              ++result.conservativeInserts;
            }
          }
        }
      }
    }
    result.storage[value] = payload;
    return payload;
  }

  Value root(Value value) {
    auto found = result.storageRoots.find(value);
    if (found != result.storageRoots.end()) return found->second;
    Operation *op = value.getDefiningOp();
    Value source;
    if (op && isa<PackFragOp, UnpackFragOp, vector::ShapeCastOp>(op))
      source = op->getOperand(0);
    Value representative = source ? root(source) : value;
    result.storageRoots[value] = representative;
    return representative;
  }

  SmallVector<Value, 0> sorted(const Values &values) {
    SmallVector<Value, 0> copy(values.begin(), values.end());
    llvm::sort(copy, [&](Value a, Value b) { return valueOrder.lookup(a) < valueOrder.lookup(b); });
    return copy;
  }

  void block(Block &block, const Values &inherited, unsigned depth) {
    unsigned blockID = result.blocks++;
    if (block.empty()) return;
    const auto *info = liveness.getLiveness(&block);
    Values candidates;
    candidates.insert(info->in().begin(), info->in().end());
    candidates.insert(info->out().begin(), info->out().end());
    for (Value arg : block.getArguments())
      if (!arg.use_empty()) candidates.insert(arg);
    for (Operation &op : block)
      for (Value value : op.getResults())
        if (!value.use_empty()) candidates.insert(value);

    // MLIR computes the CFG fixpoint and each value's last ancestor use.
    // Build death events once, instead of repeatedly calling the expensive
    // currentlyLiveValues/resolveLiveness query at every instruction.
    DenseMap<Operation *, SmallVector<Value>> deaths;
    Values active;
    for (Value value : candidates) {
      Operation *start = info->getStartOperation(value);
      if (info->isLiveIn(value) || isa<BlockArgument>(value)) active.insert(value);
      if (!info->isLiveOut(value))
        deaths[info->getEndOperation(value, start)].push_back(value);
    }

    for (Operation &op : block) {
      Values before = active;
      before.insert(inherited.begin(), inherited.end());
      for (Value value : deaths[&op]) active.remove(value);
      for (Value value : op.getResults())
        if (!value.use_empty()) active.insert(value);
      Values after = active;
      after.insert(inherited.begin(), inherited.end());

      RegisterPressurePoint point;
      point.operation = &op;
      point.block = blockID;
      point.depth = depth;
      point.liveBefore = sorted(before);
      point.liveAfter = sorted(after);
      point.before = result.measure(point.liveBefore);
      point.after = result.measure(point.liveAfter);
      point.ssaBefore = result.measureSSA(point.liveBefore);
      point.ssaAfter = result.measureSSA(point.liveAfter);
      if (op.getNumRegions()) {
        // A loop's initial operands and its final results need not coexist.
        point.overlap = point.before.words() >= point.after.words() ? point.before : point.after;
        point.ssaOverlap = point.ssaBefore.words() >= point.ssaAfter.words() ? point.ssaBefore : point.ssaAfter;
      } else {
        Values overlap = before;
        overlap.insert(after.begin(), after.end());
        point.overlap = result.measure(overlap.getArrayRef());
        point.ssaOverlap = result.measureSSA(overlap.getArrayRef());
      }
      result.pointIndices[&op] = result.points.size();
      result.points.push_back(std::move(point));
      if (!op.getNumRegions()) continue;

      // Values used after this region op occupy registers throughout its body,
      // even when the body never refers to them. Parent results do not exist yet.
      Values across = after;
      for (Value value : op.getResults()) across.remove(value);
      auto branch = dyn_cast<RegionBranchOpInterface>(&op);
      auto loop = dyn_cast<LoopLikeOpInterface>(&op);
      llvm::SetVector<Value> captures;
      getUsedValuesDefinedAbove(op.getRegions(), captures);
      if (!branch && !loop) {
        ++result.unknownRegionOps;
        across.insert(before.begin(), before.end());
      }
      for (Region &region : op.getRegions()) {
        Values regionAcross = across;
        bool repeated = loop || !branch || branch.isRepetitiveRegion(region.getRegionNumber());
        if (repeated) {
          // A lexical last use does not kill a captured invariant: the next
          // iteration needs it again. Use MLIR's region/loop interfaces.
          regionAcross.insert(captures.begin(), captures.end());
          if (loop) {
            for (auto bounds : {loop.getLoopLowerBounds(), loop.getLoopUpperBounds(), loop.getLoopSteps()})
              if (bounds)
                for (OpFoldResult bound : *bounds)
                  if (auto value = dyn_cast<Value>(bound)) regionAcross.insert(value);
          } else {
            // Unknown repetitive control: conservatively retain entry operands.
            regionAcross.insert(op.getOperands().begin(), op.getOperands().end());
          }
        }
        for (Block &child : region) this->block(child, regionAcross, depth + 1);
      }
    }
  }
};

using Ranges = SmallVector<std::pair<uint64_t, uint64_t>, 2>;
using Contributions = SmallVector<std::pair<Value, RegisterPressureUsage>>;

static void mergeRanges(Ranges &ranges) {
  llvm::sort(ranges);
  unsigned size = 0;
  for (auto range : ranges) {
    if (size && range.first <= ranges[size - 1].second)
      ranges[size - 1].second = std::max(ranges[size - 1].second, range.second);
    else ranges[size++] = range;
  }
  ranges.resize(size);
}

static Contributions contributions(const RegisterPressureResult &result,
                                    ArrayRef<Value> values) {
  llvm::MapVector<Value, Ranges> covered;
  for (Value value : values) {
    auto storage = result.storage.find(value);
    if (storage == result.storage.end()) { covered[value].push_back({0, 0}); continue; }
    for (auto slice : storage->second.slices)
      covered[slice.origin].push_back({slice.offset, slice.offset + slice.bitWidth});
  }
  for (auto &[origin, ranges] : covered) mergeRanges(ranges);
  auto contains = [&](RegisterPressureSlice slice) {
    auto cost = result.costs.find(slice.origin);
    if (cost != result.costs.end() && !cost->second.diagnosticWords() && !cost->second.unknownValues)
      return true; // constants need no persistent content
    auto found = covered.find(slice.origin);
    if (found == covered.end()) return false;
    return llvm::any_of(found->second, [&](auto range) {
      return range.first <= slice.offset && range.second >= slice.offset + slice.bitWidth;
    });
  };
  Contributions output;
  for (const auto &[origin, ranges] : covered) {
    auto subset = result.subsetSources.find(origin);
    if (subset != result.subsetSources.end()) {
      // Subset dependencies follow SSA definitions and cannot form a cycle.
      // Nested subsets can therefore be discharged transitively against this
      // union without a new control-flow dataflow solver.
      const auto &source = result.storage.find(subset->second)->second;
      if (llvm::all_of(source.slices, contains)) continue;
    }
    RegisterPressureUsage usage;
    auto found = result.costs.find(origin);
    if (found == result.costs.end()) usage.unknownValues = 1;
    else if (!result.elementBits.lookup(origin)) usage = found->second;
    else if (found->second.dataWords) {
      uint64_t bits = 0, elementBits = result.elementBits.lookup(origin);
      // Slot numbering preserves separate originating subword elements. A
      // bitcast merely reinterprets these slots; it does not allocate a copy.
      auto slot = [&](uint64_t bit) {
        return bit / elementBits * words(elementBits) + (bit % elementBits) / 32;
      };
      Ranges slots;
      for (auto [begin, end] : ranges) {
        if (begin == end) continue;
        bits += end - begin;
        slots.push_back({slot(begin), slot(end - 1) + 1});
      }
      mergeRanges(slots);
      for (auto [begin, end] : slots) usage.dataWords += end - begin;
      usage.packedDataWords = words(bits);
      usage.dataBits = bits;
    }
    if (usage.diagnosticWords() || usage.unknownValues) output.emplace_back(origin, usage);
  }
  return output;
}

static void printUsage(llvm::raw_ostream &os, const RegisterPressureUsage &usage) {
  os << usage.words() << " (data=" << usage.dataWords
     << ", packed-data=" << usage.packedDataWords
     << ", ideal-packed-data=" << usage.idealPackedDataWords()
     << ", addresses=" << usage.addressWords
     << ", descriptor-diagnostic=" << usage.descriptorWords
     << ", unknown=" << usage.unknownValues << ")";
}

static uint64_t peak(const RegisterPressurePoint &point) {
  return std::max(point.before.words(), point.after.words());
}

class AnalyzeRegisterPressurePass
    : public PassWrapper<AnalyzeRegisterPressurePass, OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AnalyzeRegisterPressurePass)
  AnalyzeRegisterPressurePass() = default;
  AnalyzeRegisterPressurePass(const AnalyzeRegisterPressurePass &other) : PassWrapper(other) {}
  explicit AnalyzeRegisterPressurePass(StringRef label) { stage = label.str(); }
  Option<std::string> stage{*this, "stage", llvm::cl::init("")};
  Option<unsigned> top{*this, "top", llvm::cl::init(8), llvm::cl::desc("Hotspots and contributors to print")};
  Option<bool> allPoints{*this, "all-points", llvm::cl::init(false)};
  StringRef getArgument() const final { return "frisk-analyze-register-pressure"; }
  StringRef getDescription() const final { return "Report whole-function SSA register pressure using MLIR liveness"; }
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<LLVM::LLVMDialect>();
  }
  void runOnOperation() final {
    if (!getOperation().isExternal()) {
      // Buffer one complete function report to avoid interleaved diagnostics
      // when MLIR schedules function passes on worker threads.
      std::string buffer;
      llvm::raw_string_ostream os(buffer);
      dumpRegisterPressure(getAnalysis<RegisterPressureAnalysis>().getResult(), os, stage, top, allPoints);
      llvm::errs() << buffer;
    }
    markAllAnalysesPreserved();
  }
};
} // namespace

Value RegisterPressureResult::storageRoot(Value value) const {
  auto found = storageRoots.find(value);
  return found == storageRoots.end() ? value : found->second;
}

RegisterPressureUsage RegisterPressureResult::measure(ArrayRef<Value> values) const {
  RegisterPressureUsage usage;
  for (auto [origin, contribution] : contributions(*this, values)) add(usage, contribution);
  return usage;
}

RegisterPressureUsage RegisterPressureResult::measureSSA(ArrayRef<Value> values) const {
  RegisterPressureUsage usage;
  llvm::SmallDenseSet<Value, 32> seen;
  for (Value value : values) {
    value = storageRoot(value);
    if (!seen.insert(value).second) continue;
    auto found = costs.find(value);
    if (found == costs.end()) ++usage.unknownValues;
    else add(usage, found->second);
  }
  return usage;
}

const RegisterPressurePoint *RegisterPressureResult::lookup(Operation *op) const {
  auto found = pointIndices.find(op);
  return found == pointIndices.end() ? nullptr : &points[found->second];
}

RegisterPressureResult analyzeRegisterPressure(func::FuncOp function, const Liveness &liveness) {
  Analyzer analyzer(function, liveness);
  for (Block &block : function.getBody()) analyzer.block(block, {}, 0);
  return std::move(analyzer.result);
}

RegisterPressureAnalysis::RegisterPressureAnalysis(Operation *operation, AnalysisManager &manager)
    : result(analyzeRegisterPressure(cast<func::FuncOp>(operation), manager.getAnalysis<Liveness>())) {}

void dumpRegisterPressure(const RegisterPressureResult &result, llvm::raw_ostream &os,
                          StringRef stage, unsigned top, bool allPoints) {
  auto function = result.function;
  os << "RegisterPressure: @" << function.getSymName() << " stage="
     << (stage.empty() ? "unspecified" : stage) << '\n';
  os << "  model=mlir-liveness units=32-bit-words/thread; not allocated VGPR/SGPR or occupancy\n"
        "  peak-live=data+explicit-pointer words; global/shared memory contents excluded\n"
        "  data=union of originating slots; packed-data=union of bits packed per origin\n"
        "  descriptor-diagnostic=full memref descriptors, excluded from pressure and hotspot ranking\n"
        "  ideal-packed-data=ceil(distinct live data bits/32), allowing packing across SSA origins; target packing constraints excluded\n"
        "  slices/inserts/bitcasts/assembly share content; unresolved inserts retain destination+input union\n"
        "  constants rematerialized; whole-value SSA lifetimes; lowering temporaries and target copy constraints excluded\n";
  SmallVector<unsigned> order;
  uint64_t peakLive = 0, peakOverlap = 0, peakData = 0, peakPacked = 0, peakDescriptors = 0;
  uint64_t peakIdealPacked = 0, peakIdealPackedOverlap = 0;
  uint64_t peakAddresses = 0, peakWithDescriptors = 0, ssaWithDescriptors = 0;
  uint64_t ssaLive = 0, ssaData = 0, ssaOverlap = 0;
  unsigned unknown = 0;
  for (auto [index, point] : llvm::enumerate(result.points)) {
    order.push_back(index);
    peakLive = std::max(peakLive, peak(point));
    peakOverlap = std::max(peakOverlap, point.overlap.words());
    peakIdealPackedOverlap = std::max(peakIdealPackedOverlap, point.overlap.idealPackedDataWords());
    ssaOverlap = std::max(ssaOverlap, point.ssaOverlap.words());
    for (auto usage : {point.ssaBefore, point.ssaAfter}) {
      ssaLive = std::max(ssaLive, usage.words());
      ssaData = std::max(ssaData, usage.dataWords);
      ssaWithDescriptors = std::max(ssaWithDescriptors, usage.diagnosticWords());
    }
    for (auto usage : {point.before, point.after}) {
      peakData = std::max(peakData, usage.dataWords);
      peakPacked = std::max(peakPacked, usage.packedDataWords);
      peakIdealPacked = std::max(peakIdealPacked, usage.idealPackedDataWords());
      peakDescriptors = std::max(peakDescriptors, usage.descriptorWords);
      peakAddresses = std::max(peakAddresses, usage.addressWords);
      peakWithDescriptors = std::max(peakWithDescriptors, usage.diagnosticWords());
      unknown = std::max(unknown, usage.unknownValues);
    }
  }
  os << "  summary: blocks=" << result.blocks << " points=" << result.points.size()
     << " peak-live=" << peakLive << " peak-overlap=" << peakOverlap
     << " peak-data=" << peakData << " peak-packed-data=" << peakPacked
     << " peak-ideal-packed-data=" << peakIdealPacked
     << " peak-ideal-packed-data-overlap=" << peakIdealPackedOverlap
     << " peak-addresses=" << peakAddresses
     << " peak-descriptor-diagnostic=" << peakDescriptors
     << " peak-with-descriptors=" << peakWithDescriptors << " max-unknown-live=" << unknown
     << " unknown-region-ops=" << result.unknownRegionOps
     << " ssa-peak-live=" << ssaLive << " ssa-peak-data=" << ssaData
     << " ssa-peak-overlap=" << ssaOverlap
     << " ssa-peak-with-descriptors=" << ssaWithDescriptors
     << " unresolved-extracts=" << result.unresolvedExtracts
     << " conservative-inserts=" << result.conservativeInserts << '\n';
  os << "  category peaks are independent; overlap conservatively includes dying inputs and new outputs\n";
  AsmState state(result.function);
  Values unmodeled;
  for (const auto &point : result.points)
    for (auto live : {ArrayRef<Value>(point.liveBefore), ArrayRef<Value>(point.liveAfter)})
      for (Value value : live) {
        value = result.storageRoot(value);
        if (result.costs.lookup(value).unknownValues) unmodeled.insert(value);
      }
  llvm::SetVector<Type> unknownTypes;
  for (Value value : unmodeled) unknownTypes.insert(value.getType());
  for (Type type : unknownTypes) {
    Value example;
    unsigned count = 0;
    for (Value value : unmodeled)
      if (value.getType() == type) { if (!example) example = value; ++count; }
    os << "  unmodeled-type: " << type << " values=" << count << " example=";
    example.printAsOperand(os, state);
    os << '\n';
  }
  auto printPoint = [&](StringRef prefix, unsigned index) {
    const auto &point = result.points[index];
    os << "  " << prefix << " #" << index << " block=" << point.block
       << " depth=" << point.depth << " op=" << point.operation->getName() << " before=";
    printUsage(os, point.before);
    os << " after="; printUsage(os, point.after);
    os << " overlap=" << point.overlap.words()
       << " ideal-packed-data-overlap=" << point.overlap.idealPackedDataWords()
       << " ssa-before=" << point.ssaBefore.words() << " ssa-after=" << point.ssaAfter.words()
       << " ssa-overlap=" << point.ssaOverlap.words()
       << " loc=" << point.operation->getLoc() << '\n';
  };
  // One peak per block makes nested scopes visible without dumping every op.
  SmallVector<unsigned> blockPeaks(result.blocks, unsigned(-1));
  for (unsigned index : order) {
    auto &best = blockPeaks[result.points[index].block];
    if (best == unsigned(-1) || peak(result.points[index]) > peak(result.points[best])) best = index;
  }
  for (unsigned index : blockPeaks) if (index != unsigned(-1)) printPoint("block-peak", index);
  llvm::stable_sort(order, [&](unsigned a, unsigned b) { return peak(result.points[a]) > peak(result.points[b]); });
  for (unsigned index : llvm::ArrayRef(order).take_front(top)) {
    printPoint("hotspot", index);
    const auto &point = result.points[index];
    const auto &live = point.before.words() >= point.after.words() ? point.liveBefore : point.liveAfter;
    auto contributors = contributions(result, live);
    llvm::stable_sort(contributors, [](const auto &a, const auto &b) {
      return a.second.words() > b.second.words();
    });
    for (auto [value, usage] : llvm::ArrayRef(contributors).take_front(top)) {
      os << "    content-origin "; value.printAsOperand(os, state);
      os << " : " << value.getType() << " words="; printUsage(os, usage);
      if (Operation *def = value.getDefiningOp()) os << " def=" << def->getName() << " loc=" << def->getLoc();
      else os << " block-argument=" << cast<BlockArgument>(value).getArgNumber();
      os << '\n';
    }
    if (contributors.size() > top) {
      os << "    other-content: origins=" << contributors.size() - top << " words=";
      RegisterPressureUsage other;
      for (auto [origin, usage] : llvm::ArrayRef(contributors).drop_front(top)) add(other, usage);
      printUsage(os, other);
      os << '\n';
    }
  }
  for (auto [index, point] : llvm::enumerate(result.points)) {
    if (isa<ReduceFragOp>(point.operation)) printPoint("reduction", index);
    if (allPoints) printPoint("point", index);
  }
  os << "EndRegisterPressure\n";
}

std::unique_ptr<Pass> createAnalyzeRegisterPressurePass(StringRef stage) {
  return std::make_unique<AnalyzeRegisterPressurePass>(stage);
}
} // namespace mlir::frisk
