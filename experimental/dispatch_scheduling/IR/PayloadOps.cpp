// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/dispatch_scheduling/IR/PayloadOps.h"

#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

namespace mlir::iree_compiler::Experimental {

LogicalResult PayloadCallOp::verifySymbolUses(
    SymbolTableCollection& symbolTables) {
  auto callee = symbolTables.lookupNearestSymbolFrom<FunctionOpInterface>(
      *this, getCalleeAttr());
  if (!callee || callee.isExternal()) {
    return emitOpError("callee must reference a defined compute function");
  }
  auto abiAttr = callee->getAttrOfType<TypeAttr>("iree_payload.abi");
  auto abi = abiAttr ? dyn_cast<FunctionType>(abiAttr.getValue()) : nullptr;
  if (!abi || abi.getNumResults() != 0 ||
      !llvm::equal(abi.getInputs(), getArguments().getTypes())) {
    return emitOpError("arguments must match the callee's void buffer ABI");
  }
  if (auto functionType = dyn_cast<FunctionType>(callee.getFunctionType())) {
    if (functionType != abi) {
      return emitOpError(
          "unconverted compute signature must match its buffer ABI");
    }
  }
  if (llvm::any_of(abi.getInputs(),
                   [](Type type) { return isa<TensorType>(type); })) {
    return emitOpError("compute ABI cannot contain tensor arguments");
  }
  return success();
}

LogicalResult PayloadYieldOp::verify() {
  auto parentOp = dyn_cast_or_null<PayloadRegionOp>((*this)->getParentOp());
  if (!parentOp) {
    return emitOpError("must terminate a iree_payload.region");
  }

  if (getValues().size() != parentOp.getNumResults()) {
    return emitOpError("operand count must match parent result count");
  }

  for (auto [yielded, result] :
       llvm::zip_equal(getValues(), parentOp.getResults())) {
    if (yielded.getType() != result.getType()) {
      return emitOpError("operand type must match parent result type");
    }
  }

  return success();
}

LogicalResult PayloadRegionOp::verify() {
  Block& block = getBody().front();
  Operation* terminator = block.empty() ? nullptr : &block.back();
  if (!terminator || !llvm::isa<PayloadYieldOp>(terminator)) {
    return emitOpError("requires body to terminate with iree_payload.yield");
  }

  bool hasTensorOperand = false;
  bool hasMemrefOperand = false;
  for (Value operand : getOperands()) {
    hasTensorOperand |= isa<TensorType>(operand.getType());
    hasMemrefOperand |= isa<BaseMemRefType>(operand.getType());
  }
  if (hasTensorOperand && hasMemrefOperand) {
    return emitOpError("requires pure tensor or pure buffer operands");
  }

  for (Value output : getOutputs()) {
    if (!isa<TensorType, BaseMemRefType>(output.getType())) {
      return emitOpError("outputs must be tensors or memrefs");
    }
  }

  unsigned numTensorOutputs = llvm::count_if(getOutputs(), [](Value output) {
    return isa<TensorType>(output.getType());
  });
  if (getNumResults() != numTensorOutputs) {
    return emitOpError("result count must match tensor output count");
  }

  unsigned resultIndex = 0;
  for (Value output : getOutputs()) {
    if (!isa<TensorType>(output.getType())) {
      continue;
    }
    if (output.getType() != getResult(resultIndex).getType()) {
      return emitOpError("tensor output type must match tied result type");
    }
    ++resultIndex;
  }

  unsigned expectedBlockArgs = getInputs().size() + getOutputs().size();
  if (block.getNumArguments() != expectedBlockArgs) {
    return emitOpError("entry block argument count must match ins and outs");
  }

  SmallVector<Value> operands;
  operands.append(getInputs().begin(), getInputs().end());
  operands.append(getOutputs().begin(), getOutputs().end());
  for (auto [blockArg, operand] :
       llvm::zip_equal(block.getArguments(), operands)) {
    if (blockArg.getType() != operand.getType()) {
      return emitOpError("entry block argument types must match ins and outs");
    }
  }

  return success();
}

}  // namespace mlir::iree_compiler::Experimental

#define GET_OP_CLASSES
#include "experimental/dispatch_scheduling/IR/PayloadOps.cpp.inc"
