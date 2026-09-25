// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/dispatch_scheduling/LLVMCPU/Passes.h"

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"

namespace mlir::iree_compiler::Experimental {
#define GEN_PASS_REGISTRATION
#include "experimental/dispatch_scheduling/LLVMCPU/Passes.h.inc"

void registerPayloadLLVMCPUPasses() {
  // The experiment drives these upstream conversions explicitly from scripts.
  registerPass([] { return createArithToLLVMConversionPass(); });
  registerPass([] { return createConvertControlFlowToLLVMPass(); });
  registerPass([] { return createConvertFuncToLLVMPass(); });
  registerPass([] { return createFinalizeMemRefToLLVMConversionPass(); });
  registerLLVMCPUPreparePayloadCodegenPass();
  registerLLVMCPUVerifyPayloadCodegenPass();
}
}  // namespace mlir::iree_compiler::Experimental
