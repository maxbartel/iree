// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_EXPERIMENTAL_DISPATCH_SCHEDULING_PAYLOAD_OPS
#define IREE_EXPERIMENTAL_DISPATCH_SCHEDULING_PAYLOAD_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Payload_Dialect : Dialect {
  let name = "iree_payload";
  let cppNamespace = "::mlir::iree_compiler::Experimental";
  let summary = "Experimental isolated compute payloads";
}

def Payload_PayloadYieldOp : Op<Payload_Dialect, "yield", [
    Pure, ReturnLike, Terminator]> {
  let summary = "terminator for a codegen payload region";
  let arguments = (ins Variadic<AnyType>:$values);
  let hasVerifier = 1;
  let assemblyFormat = "($values^ `:` type($values))? prop-dict attr-dict";
}

def Payload_PayloadRegionOp : Op<Payload_Dialect, "region", [
    AttrSizedOperandSegments, DestinationStyleOpInterface,
    IsolatedFromAbove, SingleBlock, SymbolTable]> {
  let summary = "isolated DPS boundary for independently scheduled compute";
  let description = [{
    An isolated compute region with explicit inputs and destination operands.
    Entry arguments match the operands, in input/destination order. Tensor
    destinations tie one-to-one to results; `payload.yield` supplies their new
    values. Buffer destinations are updated in place and have no results.
    Inputs are read-only; distinct operands do not imply non-aliasing buffers.
    Scalar inputs carry captured dimensions and other explicit context.

    Bufferization retains the wrapper, converts its shaped arguments and
    results, and materializes yielded tensor values in their destinations.
    Scheduling may then lower the body independently of its surrounding tensor
    program. The wrapper must be explicitly finalized before target export.

    The region owns a symbol table. An outlined body consists of a nested
    compute module, a `payload.call` to its designated entry, and the yield.
    The function inside that module may have arbitrary control flow and a
    converted signature while the outer DPS boundary remains unchanged.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs, Variadic<AnyShaped>:$outputs);
  let results = (outs Variadic<AnyRankedTensor>:$results);
  let regions = (region SizedRegion<1>:$body);
  let hasVerifier = 1;
  let assemblyFormat = [{
    (`ins` `(` $inputs^ `:` type($inputs) `)`)?
    (`outs` `(` $outputs^ `:` type($outputs) `)`)?
    $body (`->` type($results)^)? prop-dict attr-dict
  }];
  let extraClassDeclaration = [{
    MutableOperandRange getDpsInitsMutable() { return getOutputsMutable(); }
  }];
}

def Payload_PayloadCallOp : Op<Payload_Dialect, "call", [
    DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "invoke owned compute through its unconverted buffer ABI";
  let description = [{
    Invokes the designated compute entry owned by a payload. The entry's
    `iree_payload.abi` function type records the buffer/scalar signature
    before target conversion. The operands retain that signature while the
    callee can already have a converted target signature. Target lowering must
    explicitly materialize the corresponding descriptors and scalar arguments;
    an unresolved conversion cast is not an invocation ABI.

    The callee has no results: destinations are explicit buffer arguments.
    This operation conservatively has unknown effects until target finalization.
    It is eliminated together with the payload boundary before export.
  }];
  let arguments = (ins SymbolRefAttr:$callee, Variadic<AnyType>:$arguments);
  let assemblyFormat = "$callee `(` $arguments `)` prop-dict attr-dict `:` functional-type($arguments, results)";
}

#endif
