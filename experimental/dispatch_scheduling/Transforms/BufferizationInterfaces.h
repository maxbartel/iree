// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_EXPERIMENTAL_DISPATCH_SCHEDULING_BUFFERIZATION_INTERFACES_H_
#define IREE_EXPERIMENTAL_DISPATCH_SCHEDULING_BUFFERIZATION_INTERFACES_H_

namespace mlir {
class DialectRegistry;
class Value;
}  // namespace mlir

namespace mlir::iree_compiler::Experimental {
void registerPayloadBufferizationInterfaces(DialectRegistry& registry);

// Prove descriptor identity before constructing a destination copy. Aliasing
// storage alone is insufficient: offsets, sizes, strides and element order
// must agree, including through adapters and unchanged loop-carried values.
bool areIdenticalPayloadBufferViews(Value lhs, Value rhs);
}  // namespace mlir::iree_compiler::Experimental

#endif  // IREE_EXPERIMENTAL_DISPATCH_SCHEDULING_BUFFERIZATION_INTERFACES_H_
