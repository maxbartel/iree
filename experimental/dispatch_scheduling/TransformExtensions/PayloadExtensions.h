// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_EXPERIMENTAL_DISPATCH_SCHEDULING_TRANSFORM_OPS_H_
#define IREE_EXPERIMENTAL_DISPATCH_SCHEDULING_TRANSFORM_OPS_H_

#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/Interfaces/MatchInterfaces.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"

#define GET_OP_CLASSES
#include "experimental/dispatch_scheduling/TransformExtensions/PayloadExtensionsOps.h.inc"

namespace mlir::iree_compiler::Experimental {
void registerPayloadTransformExtension(DialectRegistry& registry);
}  // namespace mlir::iree_compiler::Experimental

#endif  // IREE_EXPERIMENTAL_DISPATCH_SCHEDULING_TRANSFORM_OPS_H_
