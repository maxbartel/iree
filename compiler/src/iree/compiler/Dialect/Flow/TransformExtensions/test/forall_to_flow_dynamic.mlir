// RUN: iree-opt %s --iree-transform-dialect-interpreter --iree-transform-dialect-drop-schedule -o %t.mlir
// RUN: FileCheck %s < %t.mlir
// RUN: iree-compile %t.mlir --iree-hal-target-backends=llvm-cpu --iree-llvmcpu-target-cpu=generic --iree-llvmcpu-target-triple=aarch64-unknown-unknown-eabi-elf -o %t.vmfb

// The dynamic dimensions are at positions 1 and 3, not 0 and 1. Both the
// metadata and body captures must be complete to survive Flow/Stream/HAL.
// Native check: @interleaved(3, 5, 2x3x3x5xf32=3) produces 2x3x3x5xf32=4.
// CHECK-LABEL: func.func @interleaved(
// CHECK-SAME: %[[M:[a-zA-Z0-9_]+]]: index, %[[N:[a-zA-Z0-9_]+]]: index, %[[INIT:[a-zA-Z0-9_]+]]: tensor<2x?x3x?xf32>
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[DM:.*]] = tensor.dim %[[INIT]], %[[C1]]
// CHECK: %[[C3:.*]] = arith.constant 3 : index
// CHECK: %[[DN:.*]] = tensor.dim %[[INIT]], %[[C3]]
// CHECK: flow.dispatch.workgroups[%[[M]], %[[N]]](%[[INIT]], %[[DM]], %[[DN]]) : (tensor<2x?x3x?xf32>{%[[DM]], %[[DN]]}, index, index)
// CHECK: (%[[BUFFER:[a-zA-Z0-9_]+]]: !iree_tensor_ext.dispatch.tensor<readwrite:tensor<2x?x3x?xf32>>, %[[BM:[a-zA-Z0-9_]+]]: index, %[[BN:[a-zA-Z0-9_]+]]: index)
// CHECK: %[[I:.*]] = flow.dispatch.workgroup.id[0]
// CHECK: %[[J:.*]] = flow.dispatch.workgroup.id[1]
// CHECK: iree_tensor_ext.dispatch.tensor.load %[[BUFFER]], offsets = [0, %[[I]], 0, %[[J]]], sizes = [2, 1, 3, 1]
// CHECK-SAME: {%[[BM]], %[[BN]]}
// CHECK: arith.addf
// CHECK: iree_tensor_ext.dispatch.tensor.store %{{.*}}, %[[BUFFER]], offsets = [0, %[[I]], 0, %[[J]]], sizes = [2, 1, 3, 1]
// CHECK-SAME: {%[[BM]], %[[BN]]}
// CHECK: count(%[[WM:[a-zA-Z0-9_]+]]: index, %[[WN:[a-zA-Z0-9_]+]]: index)
// CHECK: %[[ONE:.*]] = arith.constant 1 : index
// CHECK: flow.return %[[WM]], %[[WN]], %[[ONE]] : index, index, index

module {
  func.func @interleaved(%m: index, %n: index, %init: tensor<2x?x3x?xf32>) -> tensor<2x?x3x?xf32> {
    %result = scf.forall (%i, %j) in (%m, %n) shared_outs(%out = %init) -> tensor<2x?x3x?xf32> {
      %slice = tensor.extract_slice %out[0, %i, 0, %j] [2, 1, 3, 1] [1, 1, 1, 1] : tensor<2x?x3x?xf32> to tensor<2x1x3x1xf32>
      %one = arith.constant 1.0 : f32
      %sum = linalg.generic {
        indexing_maps = [affine_map<(i,j,k,l)->(i,j,k,l)>],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]
      } outs(%slice : tensor<2x1x3x1xf32>) {
      ^bb0(%value: f32):
        %updated = arith.addf %value, %one : f32
        linalg.yield %updated : f32
      } -> tensor<2x1x3x1xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %sum into %out[0, %i, 0, %j] [2, 1, 3, 1] [1, 1, 1, 1] : tensor<2x1x3x1xf32> into tensor<2x?x3x?xf32>
      }
    }
    return %result : tensor<2x?x3x?xf32>
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
      %loops = transform.structured.match ops{["scf.forall"]} in %root : (!transform.any_op) -> !transform.any_op
      %dispatches = transform.iree.forall_to_flow %loops : (!transform.any_op) -> !transform.any_op
      transform.yield
    }
  }
}
