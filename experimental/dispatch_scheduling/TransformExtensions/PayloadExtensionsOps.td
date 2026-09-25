// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_EXPERIMENTAL_DISPATCH_SCHEDULING_TRANSFORM_OPS
#define IREE_EXPERIMENTAL_DISPATCH_SCHEDULING_TRANSFORM_OPS

include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.td"
include "mlir/Dialect/Transform/Interfaces/MatchInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

def LowerPayloadRegionOp : Op<Transform_Dialect,
    "iree.lower_payload_region",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     TransformEachOpTrait,
     TransformOpInterface,
     ReportTrackingListenerFailuresOpTrait]> {
  let description = [{
    Lowers a `iree_payload.region` op by inlining its body at the op's
    site and erasing the wrapper. Block arguments are remapped to the
    region's `ins` and `outs` operands so the inlined body is well
    defined at the parent scope.

    For tensor-DPS form (the op has tensor results), the values yielded
    by the `iree_payload.yield` terminator replace the op's results.
    For memref-DPS form (no tensor results), the op is erased outright.

    Op-agnostic over the body content: the caller is responsible for
    ensuring the body's ops are legal at the parent scope. Bufferization
    or any other inside-region codegen should run before this transform
    so the body matches the surrounding IR's value category.

    Consumes the target handle.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target);
  let results = (outs);
  let assemblyFormat = "$target prop-dict attr-dict `:` type($target)";
  let cppNamespace =
      "mlir::iree_compiler::Experimental::transform_dialect";
  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::Operation *target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

def ReplaceLinalgInitWithEmptyOp : Op<Transform_Dialect,
    "iree.replace_linalg_init_with_empty",
    [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
     TransformEachOpTrait, TransformOpInterface,
     ReportTrackingListenerFailuresOpTrait]> {
  let summary = "Detach a fully overwritten tensor destination from old storage";
  let description = [{
    Replace the selected DPS init with tensor.empty of the same shape and
    encoding. Require pure tensor Linalg semantics, all parallel iterators,
    an unused destination block argument, and a permutation output map.
    These conditions prove that every output element is overwritten without
    reading its previous contents. Reduction and partial-write operations are
    unsupported, even if their output block argument is unused.

    Dynamic sizes are queried from the original init; other uses are preserved.
    This changes tensor dataflow before tiling and bufferization. It does not
    erase copies or assert that a borrowed input can be overwritten in place.
    An existing single-use tensor.empty destination is left unchanged. A
    shared empty destination is replaced with a fresh one, so consumer fusion
    can give independent results distinct scf.forall shared outputs.

    Consumes the target handle and returns the updated Linalg operations.
    Unsupported operations produce a silenceable failure before rewriting.
  }];
  let arguments = (ins TransformHandleTypeInterface:$target,
                       DefaultValuedAttr<I64Attr, "0">:$output);
  let results = (outs TransformHandleTypeInterface:$result);
  let assemblyFormat =
      "$target prop-dict attr-dict `:` functional-type(operands, results)";
  let cppNamespace = "mlir::iree_compiler::Experimental::transform_dialect";
  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::Operation *target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

def ReuseLinalgInputAsInitOp : Op<Transform_Dialect,
    "iree.reuse_linalg_input_as_init",
    [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
     TransformEachOpTrait, TransformOpInterface,
     ReportTrackingListenerFailuresOpTrait]> {
  let summary = "Reuse an explicitly selected pointwise input as its destination";
  let description = [{
    Change one unused destination of a pure tensor linalg.generic to the
    selected input. This restores a destination chain before tensor tiling and
    fusion, for example from a contraction through a pointwise epilogue whose
    original destination is tensor.empty. The script chooses the input and
    output by their DPS operand indices; this op does not select fusion groups.

    Require all parallel iterators, an unused output block argument, matching
    tensor types, and identical permutation indexing maps. Dynamic dimensions
    must be proven equal by value-bounds analysis. These checks ensure that
    changing the destination cannot change values or iteration bounds. Partial
    writes and unproven dynamic shapes are unsupported.
    No buffer aliasing assumption is made: later bufferization must still
    preserve any other uses of the input tensor.

    The input is initially retained. Upstream
    transform.apply_patterns.linalg.erase_unnecessary_inputs can fold the
    duplicate input/output and remove its now-redundant operand.

    Consumes the target handle and returns the updated generic operations.
    Unsupported operations produce a silenceable failure without rewriting them.
  }];
  let arguments = (ins TransformHandleTypeInterface:$target,
                       I64Attr:$input, DefaultValuedAttr<I64Attr, "0">:$output);
  let results = (outs TransformHandleTypeInterface:$result);
  let assemblyFormat =
      "$target `[` $input `]` prop-dict attr-dict `:` functional-type(operands, results)";
  let cppNamespace = "mlir::iree_compiler::Experimental::transform_dialect";
  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::Operation *target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

def WrapInPayloadGroupOp : Op<Transform_Dialect, "iree.wrap_in_payload_group",
    [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
     DeclareOpInterfaceMethods<TransformOpInterface>,
     ReportTrackingListenerFailuresOpTrait]> {
  let summary = "Wrap an explicitly selected tensor computation in a DPS payload";
  let description = [{
    Wrap the selected pure tensor DPS operations from one block in a single
    payload.region. The script chooses the group; this transform does not
    discover or tile producers. Operations are cloned in program order. Results
    used outside the group and all results of the last operation become payload
    results, in program/result order. External users must follow the last
    selected operation. Destinations are traced through selected DPS producers
    to distinct external initial values, which become the payload's outs.
    Other external operands and region captures
    become ins. Input reads retain their own argument when a value also appears
    as a destination. Scalar constants are rematerialized inside the payload so
    early scheduling can use their values. Tensor constants remain external.

    Use this after tiling/fusion and boundary-view canonicalization to give the
    complete fused computation one payload interface. Metadata and tensor slices
    may remain outside the selected group, provided they do not use its internal
    results. Empty handles return an empty handle; multiple groups should be
    processed with transform.foreach. Invalid groups fail before rewriting.

    Consumes the group handle and returns its payload region.
  }];
  let arguments = (ins TransformHandleTypeInterface:$target);
  let results = (outs TransformHandleTypeInterface:$result);
  let assemblyFormat = "$target prop-dict attr-dict `:` functional-type(operands, results)";
  let cppNamespace = "mlir::iree_compiler::Experimental::transform_dialect";
}

def WrapInPayloadRegionOp : Op<Transform_Dialect,
    "iree.wrap_in_payload_region",
    [FunctionalStyleTransformOpTrait,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     TransformEachOpTrait,
     TransformOpInterface,
     ReportTrackingListenerFailuresOpTrait]> {
  let description = [{
    Wraps the target DPS op in a `iree_payload.region`. The region's
    operand structure mirrors the target's destination-passing contract:
    `ins` are the target's DPS inputs followed by captured SSA values,
    `outs` are the target's DPS inits,
    and the region's tensor results have the same types as the target's
    tensor results.

    The cloned target lives at the body's top level, and a
    `iree_payload.yield` terminator yields the cloned op's tensor results
    so they tie to the region's results by index.

    Op-agnostic over DPS ops with pure tensor semantics; the contract
    relies on `DestinationStyleOpInterface`, not on the specific op kind.

    Processes every operation in the handle; an empty handle is a no-op.
    Consumes the `target` handle and returns two corresponding handles:
      * `$wrapped_op` -- the clone of the target op now inside the
        region body. Note that this is a different `Operation *` than
        the original target.
      * `$region` -- the newly created `iree_payload.region` op.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target);
  let results = (outs TransformHandleTypeInterface:$wrapped_op,
                      TransformHandleTypeInterface:$region);

  let assemblyFormat =
      "$target prop-dict attr-dict `:` functional-type(operands, results)";
  let cppNamespace =
      "mlir::iree_compiler::Experimental::transform_dialect";
  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::Operation *target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

def OutlinePayloadRegionOp : Op<Transform_Dialect,
    "iree.outline_payload_region", [
      DeclareOpInterfaceMethods<MemoryEffectsOpInterface>, TransformEachOpTrait,
      TransformOpInterface, ReportTrackingListenerFailuresOpTrait]> {
  let summary = "move a buffer payload body into an owned compute module";
  let description = [{
    Requires a buffer-form payload. Creates a compute module and a void entry
    function with the payload's exact buffer/scalar signature, and replaces the
    body with a `payload.call`. Symbols directly owned by the payload move into
    the compute module with the body. The payload's DPS operands are unchanged.

    The returned handle contains each owned module, which can be scheduled and
    converted using ordinary function/module passes. Consumes the payload handle;
    an empty handle is a no-op. Already outlined payloads fail before mutation.
  }];
  let arguments = (ins TransformHandleTypeInterface:$target);
  let results = (outs TransformHandleTypeInterface:$compute);
  let assemblyFormat = "$target prop-dict attr-dict `:` functional-type(operands, results)";
  let cppNamespace = "mlir::iree_compiler::Experimental::transform_dialect";
  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::Operation *target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

def BufferizePayloadBoundariesOp : Op<Transform_Dialect,
    "iree.bufferize_payload_boundaries",
    [FunctionalStyleTransformOpTrait, TransformEachOpTrait,
     TransformOpInterface, MemoryEffectsOpInterface]> {
  let summary = "Bufferize payload interfaces and optionally their computation";
  let description = [{
    Analyze each isolated, non-symbol-table scope (for example a function or
    flow.dispatch.workgroups), resolve tensor aliasing conflicts, and bufferize
    its payload.region interfaces. The computation inside each payload retains
    its tensor operations, so it can be outlined into a memref-ABI function and
    scheduled using ordinary function passes before full bufferization.

    With `bufferize_body`, bufferize the scheduled tensor computation and its
    interface together. Use this after tensor tiling, fusion and vectorization:
    body bufferization establishes actual destination views before the interface
    is rewritten. The surrounding dispatch remains in tensor form. This mode
    does not run scheduling passes or erase copies after bufferization.

    Analysis includes the computation and other uses in the scope. Filtering
    analysis down to only the payload interface would lose destination aliases
    through the tensor computation and introduce unnecessary copies. Required
    copies preserve old tensor values and simultaneous destination writes.

    Nested payloads and symbol tables are unsupported. Scopes must not overlap.
    The transform consumes and returns the scopes, invalidating handles into
    their bodies. Empty handles are harmless; scopes without payloads are
    unchanged. Failure after conflict resolution is definite.
  }];
  let arguments = (ins TransformHandleTypeInterface:$target,
                       UnitAttr:$bufferize_body);
  let results = (outs TransformHandleTypeInterface:$result);
  let assemblyFormat = "$target prop-dict attr-dict `:` functional-type(operands, results)";
  let cppNamespace = "mlir::iree_compiler::Experimental::transform_dialect";
  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::Operation *target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

def MatchMmt4dFusionRootsOp : Op<Transform_Dialect,
    "iree.cpu.match_mmt4d_fusion_roots",
    [MatchOpInterface, SingleOpMatcher, MemoryEffectsOpInterface]> {
  let summary = "Select disjoint mmt4d and pointwise groups in a function";
  let description = [{
    Read-only preflight for the experimental CPU mmt4d schedule. Return the mmt4d
    roots whose initializer and single pointwise consumer form a supported group.
    The contraction is input `contraction_input` of the epilogue. The matcher
    anchors on the enclosing function, so actions may rewrite all three sibling
    operations without escaping their mutation scope. Only direct children of
    function blocks are considered; initializers and existing dispatch bodies
    are excluded. Multi-use contraction results are excluded, giving each
    contraction and epilogue one owner. A shared initializer may be cloned by
    producer fusion; this does not duplicate weights. Unsupported operations
    are left untouched. An empty result produces a silenceable match failure.

    The schedule supports f32 tensors with fixed positive inner tiles and
    static or dynamic outer dimensions on a resolved CPU target with the mmt4d
    microkernel enabled. Epilogues
    may contain scalar constants, f32 add/subtract/multiply/negate, maximum and
    minimum. Initializers may be zero fills or single-input f32 broadcasts with
    an identity output map. Broadcast values initialize the accumulator before
    the reduction; they are not moved to a trailing addition.
    Other bodies retain classic lowering. Input and destination
    dimensions must be provably equal before retie and fusion.
  }];
  let arguments = (ins TransformHandleTypeInterface:$operand_handle,
                       I64Attr:$contraction_input);
  let results = (outs TransformHandleTypeInterface:$roots);
  let assemblyFormat = "$operand_handle prop-dict attr-dict `:` functional-type(operands, results)";
  let extraClassDeclaration = SingleOpMatcher.extraDeclaration;
  let cppNamespace = "mlir::iree_compiler::Experimental::transform_dialect";
}

def GetMmt4dWorkgroupTilesOp : Op<Transform_Dialect,
    "iree.cpu.get_mmt4d_workgroup_tiles",
    [ParamProducerTransformOpTrait, TransformEachOpTrait,
     TransformOpInterface, MemoryEffectsOpInterface]> {
  let summary = "Query the classic CPU mmt4d distribution heuristic";
  let description = [{
    Return the M and N outer tile sizes chosen by the existing CPU heuristic
    for each unconfigured linalg.mmt4d. Query before tiling, while the root has
    its original dimensions. The enclosing scope must have a resolved LLVM CPU
    target. This transform does not modify the root or attach a configuration.
    Empty handles yield empty parameters; multi-target handles retain order.
  }];
  let arguments = (ins TransformHandleTypeInterface:$target);
  let results = (outs TransformParamTypeInterface:$m, TransformParamTypeInterface:$n);
  let assemblyFormat = "$target prop-dict attr-dict `:` functional-type(operands, results)";
  let cppNamespace = "mlir::iree_compiler::Experimental::transform_dialect";
  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::Operation *target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

#endif
