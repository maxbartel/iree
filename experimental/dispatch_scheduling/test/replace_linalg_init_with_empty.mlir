// RUN: iree-opt --iree-plugin=dispatch_scheduling %s --transform-interpreter --iree-transform-dialect-drop-schedule | FileCheck %s

// Preserve shape and old-value uses; only the selected unread init changes.
// CHECK-LABEL: func.func @static_fill(
// CHECK-SAME: %[[OLD:[^ ,:]+]]:
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<4x8xf32>
// CHECK-NEXT: %[[RESULT:.*]] = linalg.fill
// CHECK-SAME: outs(%[[EMPTY]] : tensor<4x8xf32>)
// CHECK-NEXT: return %[[RESULT]], %[[OLD]]
func.func @static_fill(%old: tensor<4x8xf32>) -> (tensor<4x8xf32>, tensor<4x8xf32>) {
  %zero = arith.constant 0.0 : f32
  %r = linalg.fill ins(%zero : f32) outs(%old : tensor<4x8xf32>) -> tensor<4x8xf32>
  return %r, %old : tensor<4x8xf32>, tensor<4x8xf32>
}
// CHECK-LABEL: func.func @dynamic_fill(
// CHECK-SAME: %[[OLD:[^ ,:]+]]:
// CHECK: %[[DIM:.*]] = tensor.dim %[[OLD]],
// CHECK-NEXT: %[[EMPTY:.*]] = tensor.empty(%[[DIM]]) : tensor<?x8xf32>
// CHECK-NEXT: %[[RESULT:.*]] = linalg.fill
// CHECK-SAME: outs(%[[EMPTY]] : tensor<?x8xf32>)
func.func @dynamic_fill(%old: tensor<?x8xf32>) -> tensor<?x8xf32> {
  %zero = arith.constant 0.0 : f32
  %r = linalg.fill ins(%zero : f32) outs(%old : tensor<?x8xf32>) -> tensor<?x8xf32>
  return %r : tensor<?x8xf32>
}
// CHECK-LABEL: func.func @encoded_fill(
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<4x8xf32, "test.layout">
// CHECK-NEXT: %[[RESULT:.*]] = linalg.fill
// CHECK-SAME: outs(%[[EMPTY]] : tensor<4x8xf32, "test.layout">)
func.func @encoded_fill(%old: tensor<4x8xf32, "test.layout">) -> tensor<4x8xf32, "test.layout"> {
  %zero = arith.constant 0.0 : f32
  %r = linalg.fill ins(%zero : f32) outs(%old : tensor<4x8xf32, "test.layout">) -> tensor<4x8xf32, "test.layout">
  return %r : tensor<4x8xf32, "test.layout">
}
// Shared empties must become independent destinations before consumer fusion.
// CHECK-LABEL: func.func @shared_empty(
// CHECK: %[[OLD:.*]] = tensor.empty
// CHECK: %[[DIM1:.*]] = tensor.dim %[[OLD]],
// CHECK: %[[FIRST:.*]] = tensor.empty(%[[DIM1]])
// CHECK: linalg.fill {{.*}} outs(%[[FIRST]]
// CHECK: %[[DIM2:.*]] = tensor.dim %[[OLD]],
// CHECK: %[[SECOND:.*]] = tensor.empty(%[[DIM2]])
// CHECK: linalg.fill {{.*}} outs(%[[SECOND]]
func.func @shared_empty(%n: index) -> (tensor<?x8xf32>, tensor<?x8xf32>) {
  %zero = arith.constant 0.0 : f32
  %one = arith.constant 1.0 : f32
  %empty = tensor.empty(%n) : tensor<?x8xf32>
  %a = linalg.fill ins(%zero : f32) outs(%empty : tensor<?x8xf32>) -> tensor<?x8xf32>
  %b = linalg.fill ins(%one : f32) outs(%empty : tensor<?x8xf32>) -> tensor<?x8xf32>
  return %a, %b : tensor<?x8xf32>, tensor<?x8xf32>
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %fills = transform.structured.match ops{["linalg.fill"]} in %root
      : (!transform.any_op) -> !transform.any_op
    %fresh = transform.iree.replace_linalg_init_with_empty %fills
      : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
