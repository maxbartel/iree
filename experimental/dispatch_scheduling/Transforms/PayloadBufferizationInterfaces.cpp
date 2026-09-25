// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/dispatch_scheduling/IR/PayloadDialect.h"
#include "experimental/dispatch_scheduling/IR/PayloadOps.h"
#include "experimental/dispatch_scheduling/Transforms/BufferizationInterfaces.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"

namespace mlir::iree_compiler::Experimental {
namespace {
using namespace mlir::bufferization;

// Find an identical buffer view through adapters and unchanged loop-carried
// descriptors. Do not look through tensor computations, subviews or arbitrary
// aliases: sharing storage alone does not make a destination write redundant.
static Value getForwardedBuffer(Value value) {
  while (true) {
    if (auto cast = value.getDefiningOp<memref::CastOp>()) {
      value = cast.getSource();
      continue;
    }
    if (auto toBuffer = value.getDefiningOp<ToBufferOp>()) {
      if (auto toTensor = toBuffer.getTensor().getDefiningOp<ToTensorOp>()) {
        if (memref::CastOp::areCastCompatible(toTensor.getBuffer().getType(),
                                              toBuffer.getType())) {
          value = toTensor.getBuffer();
          continue;
        }
      }
    }
    if (auto result = dyn_cast<OpResult>(value)) {
      if (auto branch = dyn_cast<scf::IfOp>(result.getOwner())) {
        unsigned index = result.getResultNumber();
        Value thenBuffer =
            getForwardedBuffer(branch.thenYield().getOperand(index));
        Value elseBuffer =
            getForwardedBuffer(branch.elseYield().getOperand(index));
        // Both arms must yield the identical descriptor, not merely views
        // that may alias. Prove this before deciding to create a copy.
        if (thenBuffer == elseBuffer) {
          value = thenBuffer;
          continue;
        }
      }
      if (auto loop = dyn_cast<scf::ForOp>(result.getOwner())) {
        unsigned index = result.getResultNumber();
        Value yielded = loop.getBody()->getTerminator()->getOperand(index);
        if (getForwardedBuffer(yielded) == loop.getRegionIterArg(index)) {
          value = loop.getInitArgs()[index];
          continue;
        }
      }
    }
    return value;
  }
}

struct PayloadYieldOpInterface
    : BufferizableOpInterface::ExternalModel<PayloadYieldOpInterface,
                                             PayloadYieldOp> {
  bool bufferizesToMemoryRead(Operation*, OpOperand&,
                              const AnalysisState&) const {
    // The parent materializes each yielded value into its tied destination.
    return true;
  }
  bool bufferizesToMemoryWrite(Operation*, OpOperand&,
                               const AnalysisState&) const {
    return false;
  }
  bool mustBufferizeInPlace(Operation*, OpOperand&,
                            const AnalysisState&) const {
    return true;
  }
  AliasingValueList getAliasingValues(Operation*, OpOperand&,
                                      const AnalysisState&) const {
    // The parent advertises the DPS result alias. Linking it through the yield
    // as well would make the parent's write appear to conflict with itself.
    return {};
  }
  LogicalResult bufferize(Operation*, RewriterBase&,
                          const BufferizationOptions&,
                          BufferizationState&) const {
    // Rewritten by the parent before traversing this terminator.
    return success();
  }
};

struct PayloadRegionOpInterface
    : BufferizableOpInterface::ExternalModel<PayloadRegionOpInterface,
                                             PayloadRegionOp> {
  bool bufferizesToMemoryRead(Operation* operation, OpOperand& operand,
                              const AnalysisState&) const {
    // Tensor scheduling can make captured initializers unused. They must not
    // introduce a read/write conflict with the destination they once seeded:
    // no read remains, so constructing a defensive copy would be unnecessary.
    auto op = cast<PayloadRegionOp>(operation);
    return !op.getBody()
                .front()
                .getArgument(operand.getOperandNumber())
                .use_empty();
  }
  bool bufferizesToMemoryWrite(Operation* operation, OpOperand& operand,
                               const AnalysisState&) const {
    return cast<PayloadRegionOp>(operation).isDpsInit(&operand);
  }
  bool isWritable(Operation* operation, Value value,
                  const AnalysisState&) const {
    auto op = cast<PayloadRegionOp>(operation);
    if (auto arg = dyn_cast<BlockArgument>(value)) {
      return arg.getOwner() == &op.getBody().front() &&
             arg.getArgNumber() >= op.getInputs().size();
    }
    return value.getDefiningOp() == operation;
  }
  AliasingValueList getAliasingValues(Operation* operation, OpOperand& operand,
                                      const AnalysisState&) const {
    auto op = cast<PayloadRegionOp>(operation);
    if (!op.isDpsInit(&operand)) {
      return {};
    }
    return {{op.getTiedOpResult(&operand), BufferRelation::Equivalent}};
  }
  LogicalResult resolveConflicts(Operation* operation, RewriterBase& rewriter,
                                 const AnalysisState& analysis,
                                 const BufferizationState& state) const {
    auto bufferizable = cast<BufferizableOpInterface>(operation);
    if (failed(bufferizable.resolveTensorOpOperandConflicts(rewriter, analysis,
                                                            state))) {
      return failure();
    }
    auto op = cast<PayloadRegionOp>(operation);
    Block& block = op.getBody().front();
    auto yield = cast<PayloadYieldOp>(block.getTerminator());
    SmallVector<Value> values(yield.getValues());
    rewriter.setInsertionPoint(yield);
    for (auto item : llvm::enumerate(values)) {
      auto index = item.index();
      Value value = item.value();
      // Materialize all results simultaneously: a swapped output or a subset
      // of another destination must be read before any destination is written.
      bool needsSnapshot = llvm::any_of(
          llvm::enumerate(
              block.getArguments().drop_front(op.getInputs().size())),
          [&](auto destination) {
            return analysis.areAliasingBufferizedValues(value,
                                                        destination.value()) &&
                   (destination.index() != index ||
                    !analysis.areEquivalentBufferizedValues(
                        value, destination.value()));
          });
      if (!needsSnapshot) {
        continue;
      }
      FailureOr<Value> snapshot = allocateTensorForShapedValue(
          rewriter, yield.getLoc(), value, analysis.getOptions(), state);
      if (failed(snapshot)) {
        return failure();
      }
      values[index] = *snapshot;
    }
    rewriter.modifyOpInPlace(
        yield, [&]() { yield.getValuesMutable().assign(values); });
    return success();
  }
  FailureOr<BufferLikeType> getBufferType(
      Operation* operation, Value value, const BufferizationOptions& options,
      const BufferizationState& state,
      SmallVector<Value>& invocationStack) const {
    auto op = cast<PayloadRegionOp>(operation);
    Value operand;
    if (auto arg = dyn_cast<BlockArgument>(value)) {
      operand = op->getOperand(arg.getArgNumber());
    } else {
      operand =
          op.getDpsInitOperand(cast<OpResult>(value).getResultNumber())->get();
    }
    // Preserve actual offsets/strides/memory spaces across the isolated scope.
    return bufferization::getBufferType(operand, options, state,
                                        invocationStack);
  }
  LogicalResult bufferize(Operation* operation, RewriterBase& rewriter,
                          const BufferizationOptions& options,
                          BufferizationState& state) const {
    auto op = cast<PayloadRegionOp>(operation);
    SmallVector<Value> operands;
    for (Value operand : op->getOperands()) {
      if (!isa<TensorType>(operand.getType())) {
        operands.push_back(operand);
        continue;
      }
      FailureOr<Value> buffer = getBuffer(rewriter, operand, options, state);
      if (failed(buffer)) {
        return failure();
      }
      operands.push_back(*buffer);
    }

    unsigned numInputs = op.getInputs().size();
    auto newOp =
        PayloadRegionOp::create(rewriter, op.getLoc(), TypeRange{},
                                ValueRange(operands).take_front(numInputs),
                                ValueRange(operands).drop_front(numInputs));
    // Operand counts do not change. Preserve scheduling/target annotations as
    // well as the structural segment sizes across the tensor/buffer boundary.
    newOp->setAttrs(op->getAttrs());
    rewriter.inlineRegionBefore(op.getBody(), newOp.getBody(),
                                newOp.getBody().end());
    Block& block = newOp.getBody().front();
    for (auto [arg, operand] :
         llvm::zip_equal(block.getArguments(), operands)) {
      Type oldType = arg.getType();
      if (oldType == operand.getType()) {
        continue;
      }
      SmallVector<OpOperand*> uses;
      for (OpOperand& use : arg.getUses()) {
        uses.push_back(&use);
      }
      arg.setType(operand.getType());
      if (uses.empty()) {
        continue;
      }
      rewriter.setInsertionPointToStart(&block);
      auto tensor = ToTensorOp::create(rewriter, arg.getLoc(), oldType, arg);
      // There is exactly one tensor view per new block argument. This does not
      // add noalias attributes to the eventual compute ABI.
      tensor.setRestrict(true);
      tensor.setWritable(arg.getArgNumber() >= numInputs);
      for (OpOperand* use : uses) {
        rewriter.modifyOpInPlace(use->getOwner(),
                                 [&]() { use->set(tensor.getResult()); });
      }
    }

    auto yield = cast<PayloadYieldOp>(block.getTerminator());
    rewriter.setInsertionPoint(yield);
    for (auto [index, value] : llvm::enumerate(yield.getValues())) {
      FailureOr<Value> buffer = getBuffer(rewriter, value, options, state);
      if (failed(buffer)) {
        return failure();
      }
      Value destination = block.getArgument(numInputs + index);
      // A DPS result must end up in its promised destination even when the
      // body yielded an unrelated tensor. When body bufferization has already
      // established an identical destination view, never create a closing copy.
      if (!areIdenticalPayloadBufferViews(*buffer, destination) &&
          failed(options.memCpyFn(rewriter, yield.getLoc(), *buffer,
                                  destination))) {
        return failure();
      }
    }
    rewriter.modifyOpInPlace(yield,
                             [&]() { yield.getValuesMutable().clear(); });
    replaceOpWithBufferizedValues(rewriter, op,
                                  ValueRange(operands).drop_front(numInputs));
    return success();
  }
};
}  // namespace

// The tensor slice bufferizer requests a copy even for an in-place masked
// update. Prove descriptor identity before constructing that copy. Matching
// storage alone is insufficient: distinct offsets, sizes or strides retain
// their required write. Do not look through tensor computations or allocations.
bool areIdenticalPayloadBufferViews(Value lhs, Value rhs) {
  lhs = getForwardedBuffer(lhs);
  rhs = getForwardedBuffer(rhs);
  if (lhs == rhs) {
    return true;
  }
  // Children bufferize before their enclosing payload. At that point the
  // tensor block argument has not yet been replaced by a to_tensor adapter,
  // and separate getBuffer calls can create distinct to_buffer operations.
  // They nevertheless describe the same buffer when the tensor and complete
  // descriptor type agree.
  auto lhsBuffer = lhs.getDefiningOp<bufferization::ToBufferOp>();
  auto rhsBuffer = rhs.getDefiningOp<bufferization::ToBufferOp>();
  if (lhsBuffer && rhsBuffer && lhs.getType() == rhs.getType() &&
      lhsBuffer.getTensor() == rhsBuffer.getTensor()) {
    return true;
  }
  auto lhsSlice = lhs.getDefiningOp<memref::SubViewOp>();
  auto rhsSlice = rhs.getDefiningOp<memref::SubViewOp>();
  return lhsSlice && rhsSlice && lhs.getType() == rhs.getType() &&
         isEqualConstantIntOrValueArray(lhsSlice.getMixedOffsets(),
                                        rhsSlice.getMixedOffsets()) &&
         isEqualConstantIntOrValueArray(lhsSlice.getMixedSizes(),
                                        rhsSlice.getMixedSizes()) &&
         isEqualConstantIntOrValueArray(lhsSlice.getMixedStrides(),
                                        rhsSlice.getMixedStrides()) &&
         areIdenticalPayloadBufferViews(lhsSlice.getSource(),
                                        rhsSlice.getSource());
}

void registerPayloadBufferizationInterfaces(DialectRegistry& registry) {
  registry.addExtension(+[](MLIRContext* ctx, PayloadDialect*) {
    PayloadRegionOp::attachInterface<PayloadRegionOpInterface>(*ctx);
    PayloadYieldOp::attachInterface<PayloadYieldOpInterface>(*ctx);
  });
}
}  // namespace mlir::iree_compiler::Experimental
