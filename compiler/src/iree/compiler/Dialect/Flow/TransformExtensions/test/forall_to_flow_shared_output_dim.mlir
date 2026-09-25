// RUN: iree-opt %s --iree-transform-dialect-interpreter --iree-transform-dialect-drop-schedule | FileCheck %s

// Check that dynamic shape queries on shared outputs survive conversion.
module {
  // A dynamic shape query on the shared output must use the new tensor view,
  // not the block argument of the erased forall. The final tile may be short.
  // CHECK-LABEL: func.func @shared_output_dim(
  // CHECK: flow.dispatch.workgroups
  // CHECK: %[[WHOLE:.*]] = iree_tensor_ext.dispatch.tensor.load
  // CHECK: %[[EXTENT:.*]] = tensor.dim %[[WHOLE]]
  // CHECK: arith.subi %[[EXTENT]]
  // CHECK: %[[SIZE:.*]] = arith.minui
  // CHECK: iree_tensor_ext.dispatch.tensor.load
  // CHECK-SAME: sizes = [%[[SIZE]]]
  // CHECK: iree_tensor_ext.dispatch.tensor.store
  // CHECK-SAME: sizes = [%[[SIZE]]]
  func.func @shared_output_dim(%init: tensor<?xf32>) -> tensor<?xf32> {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %n = tensor.dim %init, %c0 : tensor<?xf32>
    %count = arith.ceildivui %n, %c4 : index
    %result = scf.forall (%i) in (%count) shared_outs(%out = %init) -> tensor<?xf32> {
      %dim = tensor.dim %out, %c0 : tensor<?xf32>
      %offset = arith.muli %i, %c4 : index
      %remaining = arith.subi %dim, %offset : index
      %size = arith.minui %remaining, %c4 : index
      %slice = tensor.extract_slice %out[%offset] [%size] [1] : tensor<?xf32> to tensor<?xf32>
      %one = arith.constant 1.0 : f32
      %filled = linalg.fill ins(%one : f32) outs(%slice : tensor<?xf32>) -> tensor<?xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %filled into %out[%offset] [%size] [1] : tensor<?xf32> into tensor<?xf32>
      }
    }
    return %result : tensor<?xf32>
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
      %loops = transform.structured.match ops{["scf.forall"]} in %root : (!transform.any_op) -> !transform.any_op
      %dispatches = transform.iree.forall_to_flow %loops : (!transform.any_op) -> !transform.any_op
      transform.yield
    }
  }
}
