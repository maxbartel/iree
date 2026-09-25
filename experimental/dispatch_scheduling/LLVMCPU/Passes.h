// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_EXPERIMENTAL_DISPATCH_SCHEDULING_LLVMCPU_PASSES_H_
#define IREE_EXPERIMENTAL_DISPATCH_SCHEDULING_LLVMCPU_PASSES_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::Experimental {
#define GEN_PASS_DECL
#include "experimental/dispatch_scheduling/LLVMCPU/Passes.h.inc"

void registerPayloadLLVMCPUPasses();
void registerPayloadLLVMCPUInterfaces(DialectRegistry& registry);
}  // namespace mlir::iree_compiler::Experimental

#endif  // IREE_EXPERIMENTAL_DISPATCH_SCHEDULING_LLVMCPU_PASSES_H_
