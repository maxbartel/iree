// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/dispatch_scheduling/IR/PayloadDialect.h"
#include "experimental/dispatch_scheduling/Transforms/BufferizationInterfaces.h"
#include "iree/compiler/PluginAPI/Client.h"

namespace mlir::iree_compiler::Experimental {
namespace {
struct DispatchSchedulingOptions {
  void bindOptions(OptionsBinder& binder) {}
};

struct DispatchSchedulingSession
    : PluginSession<DispatchSchedulingSession, DispatchSchedulingOptions> {
  void onRegisterDialects(DialectRegistry& registry) override {
    registry.insert<PayloadDialect>();
    registerPayloadBufferizationInterfaces(registry);
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
