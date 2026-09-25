// RUN: iree-opt --iree-plugin=dispatch_scheduling --transform-interpreter --split-input-file %s | FileCheck %s

// Wrap a static linalg.matmul on tensors in a iree_payload.region. The
// op is op-agnostic over DPS ops with tensor semantics: ins/outs/result
// tying come from DestinationStyleOpInterface, the body is the cloned
// matmul, and the terminator is iree_payload.yield with the matmul's
// tensor result.

// CHECK-LABEL: func.func @wrap_matmul
// CHECK-SAME:      %[[LHS:.+]]: tensor<4x4xf32>, %[[RHS:.+]]: tensor<4x4xf32>, %[[INIT:.+]]: tensor<4x4xf32>
// CHECK:         %[[R:.+]] = iree_payload.region
// CHECK-SAME:        ins(%[[LHS]], %[[RHS]] : tensor<4x4xf32>, tensor<4x4xf32>)
// CHECK-SAME:        outs(%[[INIT]] : tensor<4x4xf32>)
// CHECK-NEXT:    ^bb0(%[[LB:.+]]: tensor<4x4xf32>, %[[RB:.+]]: tensor<4x4xf32>, %[[IB:.+]]: tensor<4x4xf32>):
// CHECK-NEXT:      %[[MM:.+]] = linalg.matmul
// CHECK-SAME:        ins(%[[LB]], %[[RB]] : tensor<4x4xf32>, tensor<4x4xf32>)
// CHECK-SAME:        outs(%[[IB]] : tensor<4x4xf32>)
// CHECK-NEXT:      iree_payload.yield %[[MM]] : tensor<4x4xf32>
// CHECK-NEXT:    } -> tensor<4x4xf32>
// CHECK:         return %[[R]]
func.func @wrap_matmul(%lhs: tensor<4x4xf32>,
                       %rhs: tensor<4x4xf32>,
                       %init: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %r = linalg.matmul
      ins(%lhs, %rhs : tensor<4x4xf32>, tensor<4x4xf32>)
      outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
  return %r : tensor<4x4xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %root
        : (!transform.any_op) -> !transform.any_op
    %wrapped, %region = transform.iree.wrap_in_payload_region %matmul
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// Both result handles are valid: the `wrapped` handle points at the
// cloned matmul inside the body, and the `region` handle points at the
// new `iree_payload.region`. Match each downstream so a stale handle
// would fail interpreter apply.

// CHECK-LABEL: func.func @wrap_and_use_handles
// CHECK:         %[[R:.+]] = iree_payload.region
// CHECK:           linalg.matmul
// CHECK:           iree_payload.yield
// CHECK:         return %[[R]]
func.func @wrap_and_use_handles(%lhs: tensor<4x4xf32>,
                                 %rhs: tensor<4x4xf32>,
                                 %init: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %r = linalg.matmul
      ins(%lhs, %rhs : tensor<4x4xf32>, tensor<4x4xf32>)
      outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
  return %r : tensor<4x4xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %root
        : (!transform.any_op) -> !transform.any_op
    %wrapped, %region = transform.iree.wrap_in_payload_region %matmul
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %inner = transform.structured.match ops{["linalg.matmul"]} in %region
        : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %wrapped {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.yield
  }
}

// -----

// A multi-target handle wraps both roots, including a scalar captured inside
// the first generic. Empty wrap/lower handles are harmless.
// CHECK-LABEL: func.func @captured_scale(
// CHECK-SAME: %[[INPUT:.*]]: tensor<4xf32>, %[[OUT:.*]]: tensor<4xf32>, %[[SCALE:.*]]: f32)
// CHECK: iree_payload.region ins(%[[INPUT]], %[[SCALE]] : tensor<4xf32>, f32) outs(%[[OUT]] : tensor<4xf32>)
// CHECK-NEXT: ^bb0(%[[INARG:.*]]: tensor<4xf32>, %[[SARG:.*]]: f32, %[[OUTARG:.*]]: tensor<4xf32>):
// CHECK: linalg.generic
// CHECK-SAME: ins(%[[INARG]] : tensor<4xf32>) outs(%[[OUTARG]] : tensor<4xf32>)
// CHECK-SAME: test.wrapped
// CHECK: arith.mulf %{{.*}}, %[[SARG]]
// CHECK: iree_payload.yield
// CHECK-NEXT: } -> tensor<4xf32> {test.region}
func.func @captured_scale(%input: tensor<4xf32>, %out: tensor<4xf32>, %scale: f32) -> tensor<4xf32> {
  %r = linalg.generic {indexing_maps = [affine_map<(i)->(i)>, affine_map<(i)->(i)>], iterator_types = ["parallel"]} ins(%input : tensor<4xf32>) outs(%out : tensor<4xf32>) {
  ^bb0(%x: f32, %y: f32):
    %v = arith.mulf %x, %scale : f32
    linalg.yield %v : f32
  } -> tensor<4xf32>
  return %r : tensor<4xf32>
}

// CHECK-LABEL: func.func @same_input_output(
// CHECK-SAME: %[[INPUT:.*]]: tensor<4xf32>)
// CHECK: iree_payload.region ins(%[[INPUT]] : tensor<4xf32>) outs(%[[INPUT]] : tensor<4xf32>)
// CHECK-NEXT: ^bb0(%[[INARG:.*]]: tensor<4xf32>, %[[OUTARG:.*]]: tensor<4xf32>):
// CHECK: linalg.generic
// CHECK-SAME: ins(%[[INARG]] : tensor<4xf32>) outs(%[[OUTARG]] : tensor<4xf32>)
// CHECK-SAME: test.wrapped
// CHECK: iree_payload.yield
// CHECK-NEXT: } -> tensor<4xf32> {test.region}
func.func @same_input_output(%input: tensor<4xf32>) -> tensor<4xf32> {
  %r = linalg.generic {indexing_maps = [affine_map<(i)->(i)>, affine_map<(i)->(i)>], iterator_types = ["parallel"]} ins(%input : tensor<4xf32>) outs(%input : tensor<4xf32>) {
  ^bb0(%x: f32, %y: f32):
    %v = arith.addf %x, %y : f32
    linalg.yield %v : f32
  } -> tensor<4xf32>
  return %r : tensor<4xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %targets = transform.structured.match ops{["linalg.generic"]} in %root : (!transform.any_op) -> !transform.any_op
    %ops, %regions = transform.iree.wrap_in_payload_region %targets : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.annotate %ops "test.wrapped" : !transform.any_op
    transform.annotate %regions "test.region" : !transform.any_op
    %empty = transform.structured.match ops{["linalg.matmul"]} in %root : (!transform.any_op) -> !transform.any_op
    %empty_ops, %empty_regions = transform.iree.wrap_in_payload_region %empty : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.iree.lower_payload_region %empty_regions : !transform.any_op
    transform.yield
  }
}
