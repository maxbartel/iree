// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/dispatch_scheduling/IR/PayloadDialect.h"
#include "experimental/dispatch_scheduling/LLVMCPU/Passes.h"
#include "experimental/dispatch_scheduling/Scheduling/Passes.h"
#include "experimental/dispatch_scheduling/TransformExtensions/PayloadExtensions.h"
#include "experimental/dispatch_scheduling/Transforms/BufferizationInterfaces.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/PluginAPI/Client.h"
#include "mlir/Pass/PassManager.h"

namespace mlir::iree_compiler::Experimental {
namespace {
struct DispatchSchedulingOptions {
  bool enabled = false;
  std::string libraryFile;
  void bindOptions(OptionsBinder& binder) {
    static llvm::cl::OptionCategory category(
        "Experimental Dispatch Scheduling");
    binder.opt<bool>(
        "iree-dispatch-scheduling", enabled, llvm::cl::cat(category),
        llvm::cl::desc(
            "Schedule supported computations before dispatch creation"));
    binder.opt<std::string>(
        "iree-dispatch-scheduling-transform-spec", libraryFile,
        llvm::cl::cat(category),
        llvm::cl::desc("Override the embedded dispatch Transform library"));
  }
};

struct DispatchSchedulingSession
    : PluginSession<DispatchSchedulingSession, DispatchSchedulingOptions> {
  static void registerPasses() {
    registerPayloadLLVMCPUPasses();
    registerDispatchSchedulingPasses();
  }

  void extendDispatchSchedulingPassPipeline(
      OpPassManager& passManager) override {
    if (!options.enabled) {
      return;
    }
    SelectDispatchSchedulesPassOptions passOptions;
    passOptions.libraryFileName = options.libraryFile;
    passManager.addPass(createSelectDispatchSchedulesPass(passOptions));
  }

  void extendExecutableConfigurationPassPipeline(
      OpPassManager& passManager) override {
    // A resumed compilation may contain payloads even when selection is off.
    // Both passes leave ordinary executable functions untouched.
    auto& modulePipeline = passManager.nest<IREE::HAL::ExecutableOp>()
                               .nest<IREE::HAL::ExecutableVariantOp>()
                               .nest<ModuleOp>();
    modulePipeline.addPass(createLLVMCPUPreparePayloadBoundariesPass());
    modulePipeline.addPass(createLLVMCPUFinalizePayloadsPass());
  }

  LogicalResult onActivate() override {
    // Serialized LLVM bodies may carry only payload attributes. Keep the
    // conversion interface available even when no payload operation is parsed.
    context->getOrLoadDialect<PayloadDialect>();
    return success();
  }

  void onRegisterDialects(DialectRegistry& registry) override {
    registry.insert<PayloadDialect>();
    registerPayloadLLVMCPUInterfaces(registry);
    registerPayloadBufferizationInterfaces(registry);
    registerPayloadTransformExtension(registry);
  }
};
}  // namespace

extern "C" bool iree_register_compiler_plugin_dispatch_scheduling(
    PluginRegistrar* registrar) {
  registrar->registerPlugin<DispatchSchedulingSession>("dispatch_scheduling");
  return true;
}
}  // namespace mlir::iree_compiler::Experimental

IREE_DEFINE_COMPILER_OPTION_FLAGS(
    mlir::iree_compiler::Experimental::DispatchSchedulingOptions);
