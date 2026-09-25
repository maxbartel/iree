// RUN: iree-opt --iree-plugin=dispatch_scheduling --transform-interpreter --split-input-file %s | FileCheck %s

// Memref-DPS form: the wrapper has no results, so lowering inlines the
// body and erases the op outright. Block args remap to the ins/outs
// operands.

// CHECK-LABEL: func.func @lower_memref_payload
// CHECK-SAME:      %[[LHS:[A-Za-z0-9_]+]]: memref<4x4xf32>, %[[RHS:[A-Za-z0-9_]+]]: memref<4x4xf32>, %[[OUT:[A-Za-z0-9_]+]]: memref<4x4xf32>
// CHECK-NOT:     iree_payload.region
// CHECK-NOT:     iree_payload.yield
// CHECK:         linalg.matmul
// CHECK-SAME:      ins(%[[LHS]], %[[RHS]] : memref<4x4xf32>, memref<4x4xf32>)
// CHECK-SAME:      outs(%[[OUT]] : memref<4x4xf32>)
func.func @lower_memref_payload(%lhs: memref<4x4xf32>,
                                %rhs: memref<4x4xf32>,
                                %out: memref<4x4xf32>) {
  iree_payload.region ins(%lhs, %rhs : memref<4x4xf32>, memref<4x4xf32>)
                      outs(%out : memref<4x4xf32>) {
  ^bb0(%lb: memref<4x4xf32>, %rb: memref<4x4xf32>, %ob: memref<4x4xf32>):
    linalg.matmul ins(%lb, %rb : memref<4x4xf32>, memref<4x4xf32>)
                  outs(%ob : memref<4x4xf32>)
    iree_payload.yield
  }
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %p = transform.structured.match ops{["iree_payload.region"]} in %root
        : (!transform.any_op) -> !transform.any_op
    transform.iree.lower_payload_region %p : !transform.any_op
    transform.yield
  }
}

// -----

// Tensor-DPS form: the wrapper has a tensor result tied to the outs by
// DPS contract. Lowering inlines the body and replaces the region's
// result with the value the yield carries (i.e. the linalg.matmul's
// own tensor result).

// CHECK-LABEL: func.func @lower_tensor_payload
// CHECK-SAME:      %[[LHS:[A-Za-z0-9_]+]]: tensor<4x4xf32>, %[[RHS:[A-Za-z0-9_]+]]: tensor<4x4xf32>, %[[INIT:[A-Za-z0-9_]+]]: tensor<4x4xf32>
// CHECK-NOT:     iree_payload.region
// CHECK-NOT:     iree_payload.yield
// CHECK:         %[[MM:.+]] = linalg.matmul
// CHECK-SAME:      ins(%[[LHS]], %[[RHS]] : tensor<4x4xf32>, tensor<4x4xf32>)
// CHECK-SAME:      outs(%[[INIT]] : tensor<4x4xf32>)
// CHECK:         return %[[MM]]
func.func @lower_tensor_payload(%lhs: tensor<4x4xf32>,
                                 %rhs: tensor<4x4xf32>,
                                 %init: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %r = iree_payload.region ins(%lhs, %rhs : tensor<4x4xf32>, tensor<4x4xf32>)
                           outs(%init : tensor<4x4xf32>) {
  ^bb0(%lb: tensor<4x4xf32>, %rb: tensor<4x4xf32>, %ob: tensor<4x4xf32>):
    %m = linalg.matmul ins(%lb, %rb : tensor<4x4xf32>, tensor<4x4xf32>)
                       outs(%ob : tensor<4x4xf32>) -> tensor<4x4xf32>
    iree_payload.yield %m : tensor<4x4xf32>
  } -> tensor<4x4xf32>
  return %r : tensor<4x4xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %p = transform.structured.match ops{["iree_payload.region"]} in %root
        : (!transform.any_op) -> !transform.any_op
    transform.iree.lower_payload_region %p : !transform.any_op
    transform.yield
  }
}

// -----

// Multi-op body: the lowering preserves intra-body SSA chains. The
// scf.for yields a memref iter result that the next op consumes; after
// inlining, the same chain must hold at the parent scope.

// CHECK-LABEL: func.func @lower_multi_op_payload
// CHECK-NOT:     iree_payload.region
// CHECK:         scf.for
// CHECK:           vector.transfer_read
// CHECK:           vector.transfer_write
// CHECK:         return
func.func @lower_multi_op_payload(%lhs: memref<32x128xf32>,
                                   %rhs: memref<128x32xf32>,
                                   %out: memref<32x32xf32>) {
  iree_payload.region ins(%lhs, %rhs : memref<32x128xf32>, memref<128x32xf32>)
                      outs(%out : memref<32x32xf32>) {
  ^bb0(%lb: memref<32x128xf32>, %rb: memref<128x32xf32>, %ob: memref<32x32xf32>):
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c128 = arith.constant 128 : index
    %cst = arith.constant 0.0 : f32
    scf.for %i = %c0 to %c128 step %c8 {
      %v = vector.transfer_read %ob[%c0, %c0], %cst {in_bounds = [true, true]}
          : memref<32x32xf32>, vector<32x32xf32>
      vector.transfer_write %v, %ob[%c0, %c0] {in_bounds = [true, true]}
          : vector<32x32xf32>, memref<32x32xf32>
    }
    iree_payload.yield
  }
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %p = transform.structured.match ops{["iree_payload.region"]} in %root
        : (!transform.any_op) -> !transform.any_op
    transform.iree.lower_payload_region %p : !transform.any_op
    transform.yield
  }
}
