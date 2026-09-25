// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Encoding/Utils/Utils.h"
#include "iree/compiler/Dialect/HAL/Analysis/DeviceAnalysis.h"
#include "iree/compiler/DispatchCreation/Passes.h"

namespace mlir::iree_compiler::DispatchCreation {
#define GEN_PASS_DEF_ASSIGNDATATILINGENCODINGSPASS
#include "iree/compiler/DispatchCreation/Passes.h.inc"

namespace {
struct AssignDataTilingEncodingsPass final
    : impl::AssignDataTilingEncodingsPassBase<AssignDataTilingEncodingsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    OpPassManager pipeline(ModuleOp::getOperationName());
    buildDataTilingEncodingPassPipeline(pipeline);
    pipeline.getDependentDialects(registry);
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (Attribute layoutTarget = module->getAttr(
            IREE::Encoding::kMaterializedLayoutTargetAttrName)) {
      IREE::HAL::DeviceAnalysis analysis(module);
      if (failed(analysis.run())) {
        return signalPassFailure();
      }
      SetVector<IREE::HAL::ExecutableTargetAttr> targets;
      analysis.gatherAllExecutableTargets(targets);
      if (targets.size() != 1 || targets[0] != layoutTarget) {
        module.emitError("cannot retarget a module with materialized layouts");
        return signalPassFailure();
      }
      return;
    }
    OpPassManager pipeline(ModuleOp::getOperationName());
    buildDataTilingEncodingPassPipeline(pipeline);
    if (failed(runPipeline(pipeline, module))) {
      signalPassFailure();
    }
  }
};
} // namespace
} // namespace mlir::iree_compiler::DispatchCreation
