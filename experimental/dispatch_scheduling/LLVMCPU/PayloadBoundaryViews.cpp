// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/dispatch_scheduling/IR/PayloadOps.h"
#include "experimental/dispatch_scheduling/LLVMCPU/Passes.h"
#include "experimental/dispatch_scheduling/Transforms/BufferizationInterfaces.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::Experimental {
#define GEN_PASS_DEF_LLVMCPUPREPAREPAYLOADBOUNDARIESPASS
#include "experimental/dispatch_scheduling/LLVMCPU/Passes.h.inc"

namespace {

// These boundary views must not go through allocation-based tensor
// bufferization after their computation has already been bufferized and
// lowered.
struct FoldPayloadBufferLoad : OpRewritePattern<bufferization::ToBufferOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(bufferization::ToBufferOp op,
                                PatternRewriter& rewriter) const override {
    auto load = op.getTensor().getDefiningOp<IREE::Codegen::LoadFromBufferOp>();
    if (!load) {
      return failure();
    }
    Value source = load.getBuffer();
    if (source.getType() == op.getType()) {
      rewriter.replaceOp(op, source);
      return success();
    }
    if (!memref::CastOp::areCastCompatible(source.getType(), op.getType())) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<memref::CastOp>(op, op.getType(), source);
    return success();
  }
};

struct FoldPayloadBufferStore
    : OpRewritePattern<IREE::Codegen::StoreToBufferOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(IREE::Codegen::StoreToBufferOp op,
                                PatternRewriter& rewriter) const override {
    Value source;
    if (auto view = op.getTensor().getDefiningOp<bufferization::ToTensorOp>()) {
      source = view.getBuffer();
    } else if (auto load =
                   op.getTensor()
                       .getDefiningOp<IREE::Codegen::LoadFromBufferOp>()) {
      source = load.getBuffer();
    }
    if (!source) {
      return failure();
    }
    // An in-place result is already in its destination view. Out-of-place
    // bufferization may instead have preserved an old input in that view and
    // computed into private storage. Its final write remains necessary.
    if (!areIdenticalPayloadBufferViews(source, op.getBuffer())) {
      memref::CopyOp::create(rewriter, op.getLoc(), source, op.getBuffer());
    }
    rewriter.eraseOp(op);
    return success();
  }
};

struct FoldPayloadBufferDim : OpRewritePattern<tensor::DimOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(tensor::DimOp op,
                                PatternRewriter& rewriter) const override {
    auto load = op.getSource().getDefiningOp<IREE::Codegen::LoadFromBufferOp>();
    if (!load) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<memref::DimOp>(op, load.getBuffer(),
                                               op.getIndex());
    return success();
  }
};

// Preserve view-only boundary chains from indexed inputs. These conversions
// construct aliases of the binding, never temporary storage. The earlier
// payload bufferization has already checked input/destination interference.
struct FoldPayloadBufferExtract : OpRewritePattern<tensor::ExtractOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(tensor::ExtractOp op,
                                PatternRewriter& rewriter) const override {
    auto load = op.getTensor().getDefiningOp<IREE::Codegen::LoadFromBufferOp>();
    if (!load) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<memref::LoadOp>(op, load.getBuffer(),
                                                op.getIndices());
    return success();
  }
};

struct FoldPayloadBufferSlice : OpRewritePattern<tensor::ExtractSliceOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(tensor::ExtractSliceOp op,
                                PatternRewriter& rewriter) const override {
    auto load = op.getSource().getDefiningOp<IREE::Codegen::LoadFromBufferOp>();
    if (!load) {
      return failure();
    }
    auto type = cast<MemRefType>(memref::SubViewOp::inferRankReducedResultType(
        op.getType().getShape(), cast<MemRefType>(load.getBuffer().getType()),
        op.getMixedOffsets(), op.getMixedSizes(), op.getMixedStrides()));
    Value view = memref::SubViewOp::create(
        rewriter, op.getLoc(), type, load.getBuffer(), op.getMixedOffsets(),
        op.getMixedSizes(), op.getMixedStrides());
    rewriter.replaceOpWithNewOp<IREE::Codegen::LoadFromBufferOp>(
        op, op.getType(), view);
    return success();
  }
};

struct FoldPayloadBufferExpand : OpRewritePattern<tensor::ExpandShapeOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(tensor::ExpandShapeOp op,
                                PatternRewriter& rewriter) const override {
    auto load = op.getSrc().getDefiningOp<IREE::Codegen::LoadFromBufferOp>();
    if (!load) {
      return failure();
    }
    if (failed(memref::ExpandShapeOp::computeExpandedType(
            cast<MemRefType>(load.getBuffer().getType()),
            op.getType().getShape(), op.getReassociationIndices()))) {
      return failure();
    }
    Value view = memref::ExpandShapeOp::create(
        rewriter, op.getLoc(), op.getType().getShape(), load.getBuffer(),
        op.getReassociationIndices(), op.getMixedOutputShape());
    rewriter.replaceOpWithNewOp<IREE::Codegen::LoadFromBufferOp>(
        op, op.getType(), view);
    return success();
  }
};

class LLVMCPUPreparePayloadBoundariesPass
    : public impl::LLVMCPUPreparePayloadBoundariesPassBase<
          LLVMCPUPreparePayloadBoundariesPass> {
 public:
  using Base::Base;
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<memref::MemRefDialect>();
    createEraseHALDescriptorTypeFromMemRefPass()->getDependentDialects(
        registry);
  }
  void runOnOperation() override {
    for (auto function : getOperation().getOps<FunctionOpInterface>()) {
      SmallVector<PayloadRegionOp> payloads;
      function.walk(
          [&](PayloadRegionOp payload) { payloads.push_back(payload); });
      if (payloads.empty()) {
        continue;
      }
      for (auto payload : payloads) {
        auto modules = payload.getBody().front().getOps<ModuleOp>();
        if (!llvm::hasSingleElement(modules) ||
            !(*modules.begin())->hasAttr("iree_payload.lowered")) {
          payload.emitError(
              "expected fully lowered owned compute before boundary lowering");
          return signalPassFailure();
        }
      }
      // HAL created the binding memrefs after the earlier memory-space cleanup.
      // Normalize the CPU descriptor marker before casting to the recorded
      // views.
      OpPassManager manager(function->getName().getStringRef());
      manager.addPass(createEraseHALDescriptorTypeFromMemRefPass());
      if (failed(runPipeline(manager, function))) {
        return signalPassFailure();
      }
      RewritePatternSet patterns(&getContext());
      patterns.add<FoldPayloadBufferLoad, FoldPayloadBufferStore,
                   FoldPayloadBufferDim, FoldPayloadBufferExtract,
                   FoldPayloadBufferSlice, FoldPayloadBufferExpand>(
          &getContext());
      if (failed(applyPatternsGreedily(function, std::move(patterns)))) {
        return signalPassFailure();
      }
      // A selected dispatch has completed all numerical bufferization.
      // Remaining tensor values indicate an unsupported adapter instead of
      // permission to silently allocate/copy a new destination through the
      // classic pipeline.
      if (function
              .walk([&](Operation* op) -> WalkResult {
                if (llvm::any_of(
                        op->getOperandTypes(),
                        [](Type type) { return isa<TensorType>(type); }) ||
                    llvm::any_of(op->getResultTypes(), [](Type type) {
                      return isa<TensorType>(type);
                    })) {
                  op->emitError(
                      "unresolved tensor adapter at a lowered "
                      "payload boundary");
                  return WalkResult::interrupt();
                }
                return WalkResult::advance();
              })
              .wasInterrupted()) {
        return signalPassFailure();
      }
      (void)setTranslationInfo(
          function,
          IREE::Codegen::TranslationInfoAttr::get(
              &getContext(), IREE::Codegen::NoPipelineAttr::get(&getContext()),
              SymbolRefAttr(), /*workgroupSize=*/{},
              /*subgroupSize=*/0, DictionaryAttr()));
    }
  }
};

}  // namespace
}  // namespace mlir::iree_compiler::Experimental
