// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUTypes.h"
#include "iree/compiler/Dialect/Encoding/Utils/Utils.h"
#include "iree/compiler/Dialect/HAL/Analysis/DeviceAnalysis.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/DispatchCreation/Passes.h"
#include "iree/compiler/GlobalOptimization/Passes.h"
#include "iree/compiler/Utils/PassUtils.h"
#include "llvm/TargetParser/Triple.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"

namespace mlir::iree_compiler::GlobalOptimization {
#define GEN_PASS_DEF_EARLYDATATILINGPASS
#include "iree/compiler/GlobalOptimization/Passes.h.inc"

namespace {
static void buildEarlyDataTilingPipeline(OpPassManager &pipeline) {
  MultiOpNest<func::FuncOp, IREE::Util::FuncOp, IREE::Util::InitializerOp>(
      pipeline)
      // Expose the same producer/consumer shapes used by dispatch formation.
      // In particular, unit-batch folding may have left a collapse between a
      // named convolution and its pointwise consumer.
      .addPass(
          []() { return DispatchCreation::createBubbleUpExpandShapesPass(); })
      .addPass(DispatchCreation::createCollapseContractionDimensionsPass)
      // Normalization itself can introduce expands after a contraction.
      .addPass(DispatchCreation::createSinkReshapesPass);
  DispatchCreation::buildDataTilingEncodingPassPipeline(pipeline);
  pipeline.addPass(DispatchCreation::createPropagateDataTilingEncodingsPass());
  // Propagation can reify dynamic dimensions from encoded epilogue results.
  // Resolve them while their logical shapes are still available, before
  // materialization replaces them with padded physical shapes.
  pipeline.addPass(memref::createResolveRankedShapeTypeResultDimsPass());
  pipeline.addPass(createMaterializeHostEncodingPass());
  // Some physical encodings only reshape a tensor, such as a packed 1-D
  // broadcast bias. Resolve these while still in tensor IR so dispatch
  // creation never turns a metadata change into a separate copy dispatch.
  pipeline.addPass(createSimplifyPackUnpackPass());
}

struct EarlyDataTilingPass final
    : impl::EarlyDataTilingPassBase<EarlyDataTilingPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    OpPassManager pipeline(ModuleOp::getOperationName());
    buildEarlyDataTilingPipeline(pipeline);
    pipeline.getDependentDialects(registry);
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    IREE::HAL::DeviceAnalysis analysis(module);
    if (failed(analysis.run())) {
      return signalPassFailure();
    }
    SetVector<IREE::HAL::ExecutableTargetAttr> targets;
    analysis.gatherAllExecutableTargets(targets);
    if (Attribute layoutTarget = module->getAttr(
            IREE::Encoding::kMaterializedLayoutTargetAttrName)) {
      if (targets.size() != 1 || targets[0] != layoutTarget) {
        module.emitError("cannot retarget a module with materialized layouts");
        return signalPassFailure();
      }
      return;
    }

    // An unsupported early target must retain the entire late layout route.
    // In particular, do not assign encodings and then erase them via identity
    // materialization for a heterogeneous module.
    if (targets.size() != 1 || targets[0].getBackend() != "llvm-cpu") {
      return;
    }
    DictionaryAttr config = targets[0].getConfiguration();
    if (!config || !config.getAs<IREE::CPU::CPUEncodingResolverAttr>(
                       IREE::Encoding::kEncodingResolverAttrName)) {
      return;
    }
    auto tripleAttr = config.getAs<StringAttr>("target_triple");
    if (!tripleAttr) {
      return;
    }
    llvm::Triple triple(tripleAttr.getValue());
    if (!triple.isAArch64() && triple.getArch() != llvm::Triple::x86_64) {
      return;
    }
    if (auto innerTiled = config.getAs<BoolAttr>("enable_inner_tiled")) {
      if (innerTiled.getValue()) {
        return;
      }
    }

    OpPassManager pipeline(ModuleOp::getOperationName());
    buildEarlyDataTilingPipeline(pipeline);
    if (failed(runPipeline(pipeline, module))) {
      return signalPassFailure();
    }
    module->setAttr(IREE::Encoding::kMaterializedLayoutTargetAttrName,
                    targets[0]);
  }
};
} // namespace
} // namespace mlir::iree_compiler::GlobalOptimization
