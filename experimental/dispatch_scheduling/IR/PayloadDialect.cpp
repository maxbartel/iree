// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/dispatch_scheduling/IR/PayloadDialect.h"

#include "experimental/dispatch_scheduling/IR/PayloadOps.h"
#include "mlir/IR/DialectImplementation.h"

namespace mlir::iree_compiler::Experimental {
void PayloadDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "experimental/dispatch_scheduling/IR/PayloadOps.cpp.inc"
      >();
}
}  // namespace mlir::iree_compiler::Experimental

#include "experimental/dispatch_scheduling/IR/PayloadDialect.cpp.inc"
