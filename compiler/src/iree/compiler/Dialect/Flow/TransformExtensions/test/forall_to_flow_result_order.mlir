// RUN: iree-opt %s --iree-transform-dialect-interpreter --iree-transform-dialect-drop-schedule -o %t.mlir
// RUN: FileCheck %s < %t.mlir
// RUN: iree-compile %t.mlir --iree-hal-target-backends=llvm-cpu --iree-llvmcpu-target-cpu=generic --iree-llvmcpu-target-triple=aarch64-unknown-unknown-eabi-elf --compile-to=stream -o %t.stream

// Fusion can insert parallel stores in a different order from forall results.
// Result types, tied operands, and dynamic dimensions must retain result order.
// The two dynamic lengths are independent and the element sizes differ.
// CHECK-LABEL: func.func @reversed_outputs(
// CHECK-SAME: %[[A:[a-zA-Z0-9_]+]]: tensor<?xf16>, %[[B:[a-zA-Z0-9_]+]]: tensor<?xf32>
// CHECK: %[[N:.*]] = tensor.dim %[[A]],
// CHECK: %[[M:.*]] = tensor.dim %[[B]],
// CHECK: %[[ADIM:.*]] = tensor.dim %[[A]],
// CHECK: %[[BDIM:.*]] = tensor.dim %[[B]],
// CHECK: %[[RESULT:.*]]:2 = flow.dispatch.workgroups[%{{.*}}](%[[N]], %[[M]], %[[A]], %[[B]], %[[ADIM]], %[[BDIM]])
// CHECK-SAME: -> (%[[A]]{%[[ADIM]]}, %[[B]]{%[[BDIM]]})
// CHECK: (%[[BN:[a-zA-Z0-9_]+]]: index, %[[BM:[a-zA-Z0-9_]+]]: index, %[[BA:[a-zA-Z0-9_]+]]: !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?xf16>>, %[[BB:[a-zA-Z0-9_]+]]: !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?xf32>>, %[[DA:[a-zA-Z0-9_]+]]: index, %[[DB:[a-zA-Z0-9_]+]]: index)
// CHECK: iree_tensor_ext.dispatch.tensor.store %{{.*}}, %[[BB]], offsets = [0], sizes = [%[[BM]]], strides = [1] : tensor<?xf32> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?xf32>>{%[[DB]]}
// CHECK: iree_tensor_ext.dispatch.tensor.store %{{.*}}, %[[BA]], offsets = [0], sizes = [%[[BN]]], strides = [1] : tensor<?xf16> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?xf16>>{%[[DA]]}
// CHECK: return %[[RESULT]]#0, %[[RESULT]]#1 : tensor<?xf16>, tensor<?xf32>

module {
  func.func @reversed_outputs(%a: tensor<?xf16>, %b: tensor<?xf32>) -> (tensor<?xf16>, tensor<?xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %n = tensor.dim %a, %c0 : tensor<?xf16>
    %m = tensor.dim %b, %c0 : tensor<?xf32>
    %r:2 = scf.forall (%i) in (%c1) shared_outs(%oa = %a, %ob = %b) -> (tensor<?xf16>, tensor<?xf32>) {
      %as = tensor.extract_slice %oa[0] [%n] [1] : tensor<?xf16> to tensor<?xf16>
      %bs = tensor.extract_slice %ob[0] [%m] [1] : tensor<?xf32> to tensor<?xf32>
      %one = arith.constant 1.0 : f16
      %two = arith.constant 2.0 : f32
      %af = linalg.fill ins(%one : f16) outs(%as : tensor<?xf16>) -> tensor<?xf16>
      %bf = linalg.fill ins(%two : f32) outs(%bs : tensor<?xf32>) -> tensor<?xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %bf into %ob[0] [%m] [1] : tensor<?xf32> into tensor<?xf32>
        tensor.parallel_insert_slice %af into %oa[0] [%n] [1] : tensor<?xf16> into tensor<?xf16>
      }
    }
    return %r#0, %r#1 : tensor<?xf16>, tensor<?xf32>
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
      %loops = transform.structured.match ops{["scf.forall"]} in %root : (!transform.any_op) -> !transform.any_op
      %dispatches = transform.iree.forall_to_flow %loops : (!transform.any_op) -> !transform.any_op
      transform.yield
    }
  }
}
