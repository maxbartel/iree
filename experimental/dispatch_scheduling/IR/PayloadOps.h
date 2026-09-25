// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_EXPERIMENTAL_DISPATCH_SCHEDULING_PAYLOAD_OPS_H_
#define IREE_EXPERIMENTAL_DISPATCH_SCHEDULING_PAYLOAD_OPS_H_

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "experimental/dispatch_scheduling/IR/PayloadOps.h.inc"

#endif  // IREE_EXPERIMENTAL_DISPATCH_SCHEDULING_PAYLOAD_OPS_H_
