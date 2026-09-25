// RUN: iree-opt --iree-plugin=dispatch_scheduling %s --transform-interpreter --iree-transform-dialect-drop-schedule | FileCheck %s

// CHECK-LABEL: func.func @retie(
// CHECK-SAME: %[[INPUT:.*]]: tensor<4x8xf32>, %[[BIAS:.*]]: tensor<4x8xf32>,
// CHECK: %[[SUM:.*]] = linalg.generic
// CHECK-SAME: ins(%[[BIAS]] : tensor<4x8xf32>) outs(%[[INPUT]] : tensor<4x8xf32>)
// CHECK: ^bb0(%[[B:.*]]: f32, %[[X:.*]]: f32):
// CHECK: arith.addf %[[X]], %[[B]]
// CHECK: return %[[SUM]], %[[INPUT]]

// Keep another use of the source tensor. Retie changes no tensor values and
// makes no promise that subsequent bufferization can overwrite a live input.
func.func @retie(%input: tensor<4x8xf32>, %bias: tensor<4x8xf32>,
                 %old: tensor<4x8xf32>) -> (tensor<4x8xf32>, tensor<4x8xf32>) {
  %sum = linalg.generic {
    indexing_maps = [affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i,j)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%input, %bias : tensor<4x8xf32>, tensor<4x8xf32>)
    outs(%old : tensor<4x8xf32>) {
  ^bb0(%x: f32, %b: f32, %unused: f32):
    %sum = arith.addf %x, %b : f32
    linalg.yield %sum : f32
  } -> tensor<4x8xf32>
  return %sum, %input : tensor<4x8xf32>, tensor<4x8xf32>
}

// CHECK-LABEL: func.func @retie_dynamic(
// CHECK-SAME: %[[DYN_INPUT:[a-zA-Z0-9_]+]]: tensor<?x8xf32>, %[[DYN_BIAS:[a-zA-Z0-9_]+]]: tensor<?x8xf32>
// CHECK: %[[DYN_SUM:.*]] = linalg.generic
// CHECK-SAME: ins(%[[DYN_BIAS]] : tensor<?x8xf32>) outs(%[[DYN_INPUT]] : tensor<?x8xf32>)
// CHECK: return %[[DYN_SUM]], %[[DYN_INPUT]]

func.func @retie_dynamic(%input: tensor<?x8xf32>, %bias: tensor<?x8xf32>) -> (tensor<?x8xf32>, tensor<?x8xf32>) {
  %c0 = arith.constant 0 : index
  %m = tensor.dim %input, %c0 : tensor<?x8xf32>
  %old = tensor.empty(%m) : tensor<?x8xf32>
  %sum = linalg.generic {
    indexing_maps = [affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i,j)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%input, %bias : tensor<?x8xf32>, tensor<?x8xf32>)
    outs(%old : tensor<?x8xf32>) {
  ^bb0(%x: f32, %b: f32, %unused: f32):
    %v = arith.addf %x, %b : f32
    linalg.yield %v : f32
  } -> tensor<?x8xf32>
  return %sum, %input : tensor<?x8xf32>, tensor<?x8xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %generic = transform.structured.match ops{["linalg.generic"]} in %root
      : (!transform.any_op) -> !transform.any_op
    %retied = transform.iree.reuse_linalg_input_as_init %generic[0]
      : (!transform.any_op) -> !transform.any_op
    %parent = transform.get_parent_op %retied
      : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %parent {
      transform.apply_patterns.linalg.erase_unnecessary_inputs
    } : !transform.any_op
    %none = transform.structured.match ops{["test.absent"]} in %root
      : (!transform.any_op) -> !transform.any_op
    %empty = transform.iree.reuse_linalg_input_as_init %none[0]
      : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
