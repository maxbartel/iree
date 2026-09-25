// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_LLVMCPU_KERNELDISPATCH_H_
#define IREE_COMPILER_CODEGEN_LLVMCPU_KERNELDISPATCH_H_

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

namespace mlir::linalg {
class LinalgOp;
} // namespace mlir::linalg

namespace mlir::iree_compiler {

LogicalResult initCPULaunchConfig(FunctionOpInterface funcOp);

/// Query the existing CPU mmt4d/batch_mmt4d heuristic without configuring the
/// op. Requires a packed contraction with the same supported shapes as the
/// selector.
IREE::Codegen::LoweringConfigAttrInterface
getMmt4dLoweringConfig(linalg::LinalgOp op, DictionaryAttr targetConfig);

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_LLVMCPU_KERNELDISPATCH_H_
