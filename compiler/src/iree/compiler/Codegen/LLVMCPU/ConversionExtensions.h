// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_LLVMCPU_CONVERSIONEXTENSIONS_H_
#define IREE_COMPILER_CODEGEN_LLVMCPU_CONVERSIONEXTENSIONS_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectInterface.h"

namespace mlir {
class ConversionTarget;
class LLVMTypeConverter;
class RewritePatternSet;
} // namespace mlir

namespace mlir::iree_compiler {

// Extend CPU conversion without making the backend depend on optional dialects.
// Implementations are shared by parallel executable compilations: callbacks
// must be stateless and modify only the supplied module or conversion objects.
class LLVMCPUConversionDialectInterface
    : public DialectInterface::Base<LLVMCPUConversionDialectInterface> {
public:
  explicit LLVMCPUConversionDialectInterface(Dialect *dialect)
      : Base(dialect) {}

  // Add conversions using the backend's type converter and target data layout.
  virtual void populateConversionPatterns(LLVMTypeConverter &typeConverter,
                                          RewritePatternSet &patterns,
                                          ConversionTarget &target) const {}

  // Finish structural adaptations after type conversion, before imported calls
  // acquire the dispatch ABI. Emit a diagnostic on failure. A callback must be
  // a no-op on modules without IR belonging to its extension.
  virtual LogicalResult finalizeConversion(ModuleOp module) const {
    return success();
  }
};
} // namespace mlir::iree_compiler

MLIR_DECLARE_EXPLICIT_TYPE_ID(
    mlir::iree_compiler::LLVMCPUConversionDialectInterface)

#endif // IREE_COMPILER_CODEGEN_LLVMCPU_CONVERSIONEXTENSIONS_H_
