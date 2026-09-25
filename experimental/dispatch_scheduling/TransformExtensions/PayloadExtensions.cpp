// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/dispatch_scheduling/TransformExtensions/PayloadExtensions.h"

#include <functional>

#include "experimental/dispatch_scheduling/IR/PayloadDialect.h"
#include "experimental/dispatch_scheduling/IR/PayloadOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "mlir/Transforms/RegionUtils.h"

namespace mlir::iree_compiler::Experimental::transform_dialect {

static bool haveSameTensorShape(Value lhs, Value rhs) {
  auto lhsType = dyn_cast<RankedTensorType>(lhs.getType());
  auto rhsType = dyn_cast<RankedTensorType>(rhs.getType());
  if (!lhsType || !rhsType || lhsType.getRank() != rhsType.getRank()) {
    return false;
  }
  for (int64_t dim = 0; dim < lhsType.getRank(); ++dim) {
    if (!lhsType.isDynamicDim(dim) && !rhsType.isDynamicDim(dim)) {
      if (lhsType.getDimSize(dim) != rhsType.getDimSize(dim)) {
        return false;
      }
      continue;
    }
    FailureOr<bool> equal =
        ValueBoundsConstraintSet::areEqual({lhs, dim}, {rhs, dim});
    if (failed(equal) || !*equal) {
      return false;
    }
  }
  return true;
}

DiagnosedSilenceableFailure ReplaceLinalgInitWithEmptyOp::applyToOne(
    transform::TransformRewriter& rewriter, Operation* target,
    transform::ApplyToEachResultList& results,
    transform::TransformState& state) {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(target);
  if (!linalgOp || !linalgOp.hasPureTensorSemantics() ||
      !isMemoryEffectFree(linalgOp) ||
      linalgOp.getNumParallelLoops() != linalgOp.getNumLoops()) {
    return emitSilenceableError()
           << "expected a pure tensor Linalg op with only parallel iterators";
  }
  if (getOutput() < 0 || getOutput() >= linalgOp.getNumDpsInits()) {
    return emitSilenceableError() << "DPS output index out of range";
  }
  OpOperand* output = linalgOp.getDpsInitOperand(getOutput());
  if (linalgOp.payloadUsesValueFromOperand(output)) {
    return emitSilenceableError()
           << "destination block argument must be unused";
  }
  if (!linalgOp.getMatchingIndexingMap(output).isPermutation()) {
    return emitSilenceableError() << "expected a permutation output map";
  }
  // Distinct destinations are required when consumer fusion appends another
  // result to a forall that already uses this empty tensor as a shared output.
  // Splitting the unread tensor init creates no copy or buffer allocation.
  if (!output->get().getDefiningOp<tensor::EmptyOp>() ||
      !output->get().hasOneUse()) {
    auto type = cast<RankedTensorType>(output->get().getType());
    rewriter.setInsertionPoint(target);
    Value empty = tensor::EmptyOp::create(
        rewriter, target->getLoc(),
        tensor::getMixedSizes(rewriter, target->getLoc(), output->get()),
        type.getElementType(), type.getEncoding());
    rewriter.modifyOpInPlace(target, [&] { output->set(empty); });
  }
  results.push_back(target);
  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure ReuseLinalgInputAsInitOp::applyToOne(
    transform::TransformRewriter& rewriter, Operation* target,
    transform::ApplyToEachResultList& results,
    transform::TransformState& state) {
  auto generic = dyn_cast<linalg::GenericOp>(target);
  if (!generic || !generic.hasPureTensorSemantics() ||
      !isMemoryEffectFree(generic) ||
      generic.getNumParallelLoops() != generic.getNumLoops()) {
    return emitSilenceableError()
           << "expected a pure tensor generic with only parallel iterators";
  }
  if (getInput() < 0 || getInput() >= generic.getNumDpsInputs() ||
      getOutput() < 0 || getOutput() >= generic.getNumDpsInits()) {
    return emitSilenceableError() << "DPS input or output index out of range";
  }
  OpOperand* input = generic.getDpsInputOperand(getInput());
  OpOperand* output = generic.getDpsInitOperand(getOutput());
  if (generic.payloadUsesValueFromOperand(output)) {
    return emitSilenceableError()
           << "destination block argument must be unused";
  }
  auto type = dyn_cast<RankedTensorType>(input->get().getType());
  if (!type || type != output->get().getType()) {
    return emitSilenceableError() << "expected matching tensor types";
  }
  AffineMap map = generic.getMatchingIndexingMap(input);
  if (!map.isPermutation() || map != generic.getMatchingIndexingMap(output)) {
    return emitSilenceableError()
           << "expected identical permutation input and output indexing maps";
  }
  if (!haveSameTensorShape(input->get(), output->get())) {
    return emitSilenceableError()
           << "input and destination shapes must be provably equal";
  }

  // This only changes a tensor destination whose previous contents are unread.
  // Keep both operand uses until upstream duplicate-input cleanup. In
  // particular, do not assert in-place bufferization or erase any copies here.
  rewriter.modifyOpInPlace(generic, [&] { output->set(input->get()); });
  results.push_back(generic);
  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure LowerPayloadRegionOp::applyToOne(
    transform::TransformRewriter& rewriter, Operation* target,
    transform::ApplyToEachResultList& results,
    transform::TransformState& state) {
  auto regionOp = dyn_cast<Experimental::PayloadRegionOp>(target);
  if (!regionOp) {
    auto diag = emitSilenceableError()
                << "target must be a iree_payload.region";
    diag.attachNote(target->getLoc()) << "target op";
    return diag;
  }

  Block& body = regionOp.getBody().front();
  if (llvm::any_of(body, [](Operation& op) {
        return isa<SymbolOpInterface>(op) ||
               op.hasTrait<OpTrait::SymbolTable>() ||
               isa<Experimental::PayloadCallOp>(op);
      })) {
    return emitSilenceableError() << "payload with owned symbols requires "
                                     "target boundary finalization";
  }

  // Remap block args -> region operands. Inputs come first, then outs;
  // both PayloadRegionOp::verify and wrap_in_payload_region preserve
  // that order, so we can zip directly.
  IRMapping map;
  for (auto [blockArg, operand] :
       llvm::zip_equal(body.getArguments(), regionOp->getOperands())) {
    map.map(blockArg, operand);
  }

  // Clone every non-terminator op at the region's site. `rewriter.clone`
  // updates `map` with each op's results, so subsequent body ops see
  // their new defs and intra-body SSA chains stay intact.
  rewriter.setInsertionPoint(regionOp);
  Operation* terminator = body.getTerminator();
  for (Operation& op : body.without_terminator()) {
    rewriter.clone(op, map);
  }

  // Tensor-DPS form: the yielded values replace the op's results. The
  // PayloadRegionOp verifier requires yield arity to match the op's
  // result count, so the lookup below is well defined.
  if (regionOp->getNumResults() > 0) {
    auto yieldOp = cast<Experimental::PayloadYieldOp>(terminator);
    SmallVector<Value> replacements;
    replacements.reserve(yieldOp.getValues().size());
    for (Value yielded : yieldOp.getValues()) {
      replacements.push_back(map.lookupOrDefault(yielded));
    }
    rewriter.replaceOp(regionOp, replacements);
  } else {
    rewriter.eraseOp(regionOp);
  }

  return DiagnosedSilenceableFailure::success();
}

void LowerPayloadRegionOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance>& effects) {
  transform::consumesHandle(getTargetMutable(), effects);
  transform::modifiesPayload(effects);
}

DiagnosedSilenceableFailure WrapInPayloadGroupOp::apply(
    transform::TransformRewriter& rewriter,
    transform::TransformResults& results, transform::TransformState& state) {
  SmallVector<Operation*> ops =
      llvm::to_vector(state.getPayloadOps(getTarget()));
  if (ops.empty()) {
    results.set(getOperation()->getResult(0), {});
    return DiagnosedSilenceableFailure::success();
  }
  Block* block = ops.front()->getBlock();
  llvm::SmallPtrSet<Operation*, 8> selected;
  for (Operation* op : ops) {
    auto dps = dyn_cast<DestinationStyleOpInterface>(op);
    if (!block || op->getBlock() != block || !selected.insert(op).second ||
        !dps || !dps.hasPureTensorSemantics() || !isMemoryEffectFree(op)) {
      return emitSilenceableError()
             << "expected distinct, pure tensor DPS operations in one block";
    }
  }
  llvm::sort(ops,
             [](Operation* a, Operation* b) { return a->isBeforeInBlock(b); });
  Operation* root = ops.back();
  SmallVector<Value> escapingResults;
  for (Operation* op : ops) {
    for (Value result : op->getResults()) {
      bool escapes = op == root;
      for (Operation* user : result.getUsers()) {
        Operation* owner = block->findAncestorOpInBlock(*user);
        if (owner && selected.contains(owner)) {
          continue;
        }
        // The payload is inserted at the last selected operation. Replacing
        // an earlier external use would violate dominance (or form a cycle
        // when that use also computes an input of the selected group).
        if (!owner || !root->isBeforeInBlock(owner)) {
          return emitSilenceableError()
                 << "external group result users must follow the last "
                    "selected operation";
        }
        escapes = true;
      }
      if (escapes) {
        escapingResults.push_back(result);
      }
    }
  }

  SmallVector<Value> outputs;
  llvm::SmallDenseSet<Value> outputSet;
  for (Value result : escapingResults) {
    auto resultDps = cast<DestinationStyleOpInterface>(result.getDefiningOp());
    Value init = resultDps.getTiedOpOperand(cast<OpResult>(result))->get();
    // An epilogue may use the preceding contraction as its DPS init. The
    // external interface must tie to the beginning of that destination chain,
    // while the internal clone keeps the intermediate tensor SSA values.
    while (Operation* producer = init.getDefiningOp()) {
      if (!selected.contains(producer)) {
        break;
      }
      auto dps = cast<DestinationStyleOpInterface>(producer);
      init = dps.getTiedOpOperand(cast<OpResult>(init))->get();
    }
    if (!outputSet.insert(init).second) {
      return emitSilenceableError()
             << "payload group requires distinct external destinations";
    }
    outputs.push_back(init);
  }

  auto isInternal = [&](Value value) {
    return selected.contains(value.getDefiningOp());
  };
  llvm::SetVector<Value> inputs;
  llvm::SetVector<Operation*> rematerialized;
  std::function<void(Value)> capture = [&](Value value) {
    Operation* producer = value.getDefiningOp();
    if (auto empty = dyn_cast_or_null<tensor::EmptyOp>(producer)) {
      // An internal temporary has no incoming contents or external storage
      // contract. Keep its shape operands at the interface and create the
      // empty inside the payload, where tensor/vector scheduling can eliminate
      // it before bufferization. Root DPS destinations remain external.
      for (Value size : empty.getDynamicSizes()) {
        capture(size);
      }
      rematerialized.insert(producer);
      return;
    }
    if (producer && producer->hasTrait<OpTrait::ConstantLike>() &&
        isa<IntegerType, IndexType, FloatType>(value.getType())) {
      // Keep scalar compile-time facts available to early codegen, such as a
      // zero fill that can fold into a microkernel's accumulator flag. Tensor
      // constants stay outside: cloning weights would duplicate their storage.
      rematerialized.insert(producer);
      return;
    }
    inputs.insert(value);
  };
  for (Operation* op : ops) {
    auto dps = cast<DestinationStyleOpInterface>(op);
    for (OpOperand& operand : op->getOpOperands()) {
      Value value = operand.get();
      if (!isInternal(value) &&
          (!dps.isDpsInit(&operand) || !outputSet.contains(value))) {
        capture(value);
      }
    }
    llvm::SetVector<Value> captures;
    getUsedValuesDefinedAbove(op->getRegions(), captures);
    for (Value value : captures) {
      if (!isInternal(value)) {
        capture(value);
      }
    }
  }

  rewriter.setInsertionPoint(root);
  SmallVector<Type> resultTypes = llvm::to_vector(TypeRange(escapingResults));
  auto payload = Experimental::PayloadRegionOp::create(
      rewriter, root->getLoc(), resultTypes, inputs.getArrayRef(), outputs);
  SmallVector<Value> arguments(inputs.begin(), inputs.end());
  llvm::append_range(arguments, outputs);
  SmallVector<Type> types;
  SmallVector<Location> locations;
  for (Value value : arguments) {
    types.push_back(value.getType());
    locations.push_back(value.getLoc());
  }
  Block* body = rewriter.createBlock(&payload.getBody(), {}, types, locations);
  IRMapping mapping;
  llvm::SmallDenseMap<Value, Value> destinationArgs;
  for (auto [index, value] : llvm::enumerate(inputs)) {
    mapping.map(value, body->getArgument(index));
  }
  for (auto [index, value] : llvm::enumerate(outputs)) {
    Value arg = body->getArgument(inputs.size() + index);
    destinationArgs[value] = arg;
    if (!mapping.contains(value)) {
      mapping.map(value, arg);
    }
  }
  rewriter.setInsertionPointToStart(body);
  for (Operation* op : rematerialized) {
    rewriter.clone(*op, mapping);
  }
  for (Operation* op : ops) {
    auto dps = cast<DestinationStyleOpInterface>(op);
    Operation* clone = rewriter.clone(*op, mapping);
    // Distinguish an old input read from a write to the same external value.
    // Other uses follow the normal tensor SSA mapping through the group.
    for (OpOperand& init : dps.getDpsInitsMutable()) {
      auto it = destinationArgs.find(init.get());
      if (it != destinationArgs.end()) {
        clone->setOperand(init.getOperandNumber(), it->second);
      }
    }
  }
  SmallVector<Value> yieldedValues = llvm::map_to_vector(
      escapingResults, [&](Value value) { return mapping.lookup(value); });
  Experimental::PayloadYieldOp::create(rewriter, root->getLoc(), yieldedValues);
  for (auto [original, replacement] :
       llvm::zip_equal(escapingResults, payload->getResults())) {
    rewriter.replaceUsesWithIf(original, replacement, [&](OpOperand& use) {
      return !selected.contains(block->findAncestorOpInBlock(*use.getOwner()));
    });
  }
  for (Operation* op : llvm::reverse(ops)) {
    rewriter.eraseOp(op);
  }
  results.set(getOperation()->getResult(0), {payload});
  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure WrapInPayloadRegionOp::applyToOne(
    transform::TransformRewriter& rewriter, Operation* target,
    transform::ApplyToEachResultList& results,
    transform::TransformState& state) {
  auto dps = dyn_cast<DestinationStyleOpInterface>(target);
  if (!dps) {
    auto diag = emitSilenceableError()
                << "target must implement DestinationStyleOpInterface";
    diag.attachNote(target->getLoc()) << "target op";
    return diag;
  }
  if (!dps.hasPureTensorSemantics()) {
    auto diag = emitSilenceableError()
                << "target must have pure tensor semantics";
    diag.attachNote(target->getLoc()) << "target op";
    return diag;
  }

  SmallVector<Value> inputs(dps.getDpsInputs());
  SmallVector<Value> outputs(dps.getDpsInits());

  // The DPS contract is that all of the target's operands flow through
  // ins/inits. Reject targets that carry extra operands (e.g. index
  // computations captured by the op) so the IsolatedFromAbove region
  // does not silently leave dangling references.
  if (target->getNumOperands() != inputs.size() + outputs.size()) {
    auto diag = emitSilenceableError()
                << "target has operands outside its DPS ins/inits; cannot "
                   "wrap into an isolated payload region";
    diag.attachNote(target->getLoc()) << "target op";
    return diag;
  }

  SmallVector<Type> resultTypes;
  resultTypes.reserve(target->getNumResults());
  for (Type t : target->getResultTypes()) {
    if (!isa<RankedTensorType>(t)) {
      auto diag = emitSilenceableError()
                  << "target result must be a ranked tensor";
      diag.attachNote(target->getLoc()) << "target op";
      return diag;
    }
    resultTypes.push_back(t);
  }

  llvm::SetVector<Value> captures;
  getUsedValuesDefinedAbove(target->getRegions(), captures);
  for (Value value : captures) {
    if (!llvm::is_contained(inputs, value) &&
        !llvm::is_contained(outputs, value)) {
      inputs.push_back(value);
    }
  }

  Location loc = target->getLoc();
  rewriter.setInsertionPoint(target);

  auto regionOp = Experimental::PayloadRegionOp::create(
      rewriter, loc, /*resultTypes=*/resultTypes,
      /*inputs=*/inputs, /*outputs=*/outputs);

  Region& body = regionOp.getBody();
  SmallVector<Type> argTypes;
  SmallVector<Location> argLocs;
  argTypes.reserve(inputs.size() + outputs.size());
  argLocs.reserve(inputs.size() + outputs.size());
  for (Value v : inputs) {
    argTypes.push_back(v.getType());
    argLocs.push_back(v.getLoc());
  }
  for (Value v : outputs) {
    argTypes.push_back(v.getType());
    argLocs.push_back(v.getLoc());
  }
  Block* entry = rewriter.createBlock(&body, body.end(), argTypes, argLocs);

  IRMapping map;
  unsigned argIdx = 0;
  for (Value v : inputs) {
    map.map(v, entry->getArgument(argIdx++));
  }
  for (Value v : outputs) {
    map.map(v, entry->getArgument(argIdx++));
  }

  rewriter.setInsertionPointToStart(entry);
  Operation* clonedOp = rewriter.clone(*target, map);
  for (auto [index, operand] : llvm::enumerate(dps.getDpsInputOperands())) {
    clonedOp->setOperand(operand->getOperandNumber(),
                         entry->getArgument(index));
  }
  for (int64_t index = 0; index < dps.getNumDpsInits(); ++index) {
    clonedOp->setOperand(dps.getDpsInitOperand(index)->getOperandNumber(),
                         entry->getArgument(inputs.size() + index));
  }

  Experimental::PayloadYieldOp::create(rewriter, loc, clonedOp->getResults());

  rewriter.replaceOp(target, regionOp->getResults());

  results.push_back(clonedOp);
  results.push_back(regionOp.getOperation());
  return DiagnosedSilenceableFailure::success();
}

void WrapInPayloadRegionOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance>& effects) {
  transform::consumesHandle(getTargetMutable(), effects);
  transform::producesHandle(getOperation()->getOpResults(), effects);
  transform::modifiesPayload(effects);
}

DiagnosedSilenceableFailure OutlinePayloadRegionOp::applyToOne(
    transform::TransformRewriter& rewriter, Operation* target,
    transform::ApplyToEachResultList& results,
    transform::TransformState& state) {
  auto payload = dyn_cast<Experimental::PayloadRegionOp>(target);
  if (!payload || payload.getNumResults() != 0 ||
      llvm::any_of(payload.getOperandTypes(),
                   [](Type type) { return isa<TensorType>(type); })) {
    return emitSilenceableError() << "target must be a buffer-form payload";
  }
  Block& body = payload.getBody().front();
  if (llvm::any_of(body, [](Operation& op) {
        return isa<Experimental::PayloadCallOp>(op);
      })) {
    return emitSilenceableError() << "payload is already outlined";
  }

  SmallVector<Operation*> symbols, compute;
  for (Operation& op : body.without_terminator()) {
    (isa<SymbolOpInterface>(op) ? symbols : compute).push_back(&op);
  }
  rewriter.setInsertionPointToStart(&body);
  auto module = ModuleOp::create(rewriter, payload.getLoc(), "compute");
  module->setAttr("iree_payload.compute", rewriter.getUnitAttr());
  for (Operation* symbol : symbols) {
    rewriter.moveOpBefore(symbol, module.getBody(), module.getBody()->end());
  }
  SymbolTable computeSymbols(module);
  rewriter.setInsertionPointToEnd(module.getBody());
  auto functionType = rewriter.getFunctionType(body.getArgumentTypes(), {});
  auto entry =
      func::FuncOp::create(rewriter, payload.getLoc(), "entry", functionType);
  // The call crosses the compute module's symbol table, so the entry must be
  // visible there. Only finalization can make it private in the executable.
  entry->setAttr("iree_payload.abi", TypeAttr::get(functionType));
  entry->setAttr("iree_payload.num_inputs",
                 rewriter.getI64IntegerAttr(payload.getInputs().size()));
  computeSymbols.insert(entry);
  Block* entryBlock = entry.addEntryBlock();
  IRMapping mapping;
  for (auto [outer, inner] :
       llvm::zip_equal(body.getArguments(), entryBlock->getArguments())) {
    mapping.map(outer, inner);
  }
  rewriter.setInsertionPointToEnd(entryBlock);
  for (Operation* op : compute) {
    rewriter.clone(*op, mapping);
  }
  func::ReturnOp::create(rewriter, payload.getLoc());
  for (Operation* op : llvm::reverse(compute)) {
    rewriter.eraseOp(op);
  }
  rewriter.setInsertionPoint(body.getTerminator());
  auto callee =
      SymbolRefAttr::get(rewriter.getContext(), module.getSymName().value(),
                         {FlatSymbolRefAttr::get(entry.getSymNameAttr())});
  Experimental::PayloadCallOp::create(rewriter, payload.getLoc(), callee,
                                      body.getArguments());
  results.push_back(module.getOperation());
  return DiagnosedSilenceableFailure::success();
}

void OutlinePayloadRegionOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance>& effects) {
  transform::consumesHandle(getTargetMutable(), effects);
  transform::producesHandle(getOperation()->getOpResults(), effects);
  transform::modifiesPayload(effects);
}

}  // namespace mlir::iree_compiler::Experimental::transform_dialect

namespace mlir::iree_compiler::Experimental {
namespace {
class PayloadTransformExtension
    : public transform::TransformDialectExtension<PayloadTransformExtension> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PayloadTransformExtension)
  PayloadTransformExtension() {
    declareGeneratedDialect<PayloadDialect>();
    declareGeneratedDialect<arith::ArithDialect>();
    declareGeneratedDialect<func::FuncDialect>();
    declareGeneratedDialect<linalg::LinalgDialect>();
    declareGeneratedDialect<tensor::TensorDialect>();
    registerTransformOps<
#define GET_OP_LIST
#include "experimental/dispatch_scheduling/TransformExtensions/PayloadExtensionsOps.cpp.inc"
        >();
  }
};
}  // namespace
void registerPayloadTransformExtension(DialectRegistry& registry) {
  registry.addExtensions<PayloadTransformExtension>();
}
}  // namespace mlir::iree_compiler::Experimental

#define GET_OP_CLASSES
#include "experimental/dispatch_scheduling/TransformExtensions/PayloadExtensionsOps.cpp.inc"
