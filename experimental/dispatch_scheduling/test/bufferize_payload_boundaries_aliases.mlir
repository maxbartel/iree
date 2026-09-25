// RUN: iree-opt --iree-plugin=dispatch_scheduling %s --transform-interpreter --canonicalize -o %t
// RUN: iree-opt --iree-plugin=dispatch_scheduling %t --pass-pipeline='builtin.module(func.func(iree_payload.region(builtin.module(func.func(iree-codegen-iree-comprehensive-bufferize,canonicalize)))))' | FileCheck %s

// Required semantic copies must survive splitting interface and body
// bufferization. These cases are outside the selected copy-free M2 family.
// CHECK-LABEL: func.func @unrelated_yield(
// CHECK: func.func @entry(%[[SOURCE:[^ ,:]+]]: memref<4xf32>, %[[DEST:[^ ,:]+]]: memref<4xf32>)
// CHECK: memref.copy %[[SOURCE]], %[[DEST]]

// Both destination snapshots must precede either destination write.
// CHECK-LABEL: func.func @swapped_yields(
// CHECK: func.func @entry(%[[FIRST:[^ ,:]+]]: memref<4xf32>, %[[SECOND:[^ ,:]+]]: memref<4xf32>)
// CHECK: %[[SECOND_COPY:[^ ,:]+]] = memref.alloc()
// CHECK: linalg.generic
// CHECK-SAME: ins(%[[SECOND]] : memref<4xf32>) outs(%[[SECOND_COPY]] : memref<4xf32>)
// CHECK: %[[FIRST_COPY:[^ ,:]+]] = memref.alloc()
// CHECK: linalg.generic
// CHECK-SAME: ins(%[[FIRST]] : memref<4xf32>) outs(%[[FIRST_COPY]] : memref<4xf32>)
// CHECK: memref.copy %[[SECOND_COPY]], %[[FIRST]]
// CHECK: memref.copy %[[FIRST_COPY]], %[[SECOND]]

// CHECK-LABEL: func.func @input_output_alias(
// CHECK-SAME: %[[OLD:[^ ,:]+]]: memref<4xf32>)
// CHECK: %[[NEW:[^ ,:]+]] = memref.alloca()
// CHECK-NEXT: memref.copy %[[OLD]], %[[NEW]]
// CHECK: iree_payload.region ins(%[[OLD]] : memref<4xf32>) outs(%[[NEW]] : memref<4xf32>)
// CHECK: func.func @entry(%[[IN:[^ ,:]+]]: memref<4xf32>, %[[OUT:[^ ,:]+]]: memref<4xf32>)
// CHECK: memref.store %{{.*}}, %[[OUT]]
// CHECK-NEXT: %[[VALUE:[^ ,:]+]] = memref.load %[[IN]]
// CHECK-NEXT: memref.store %[[VALUE]], %[[OUT]]
// CHECK: memref.copy %[[NEW]], %[[OLD]]



func.func @unrelated_yield() {
  %a = bufferization.alloc_tensor() : tensor<4xf32>
  %out = bufferization.alloc_tensor() : tensor<4xf32>
  %result = iree_payload.region ins(%a : tensor<4xf32>) outs(%out : tensor<4xf32>) {
  ^bb0(%source: tensor<4xf32>, %dest: tensor<4xf32>):
    iree_payload.yield %source : tensor<4xf32>
  } -> tensor<4xf32>
  return
}

func.func @swapped_yields() {
  %a = bufferization.alloc_tensor() : tensor<4xf32>
  %b = bufferization.alloc_tensor() : tensor<4xf32>
  %r:2 = iree_payload.region outs(%a, %b : tensor<4xf32>, tensor<4xf32>) {
  ^bb0(%first: tensor<4xf32>, %second: tensor<4xf32>):
    iree_payload.yield %second, %first : tensor<4xf32>, tensor<4xf32>
  } -> tensor<4xf32>, tensor<4xf32>
  return
}

func.func @input_output_alias(%buffer: memref<4xf32>) {
  %tensor = bufferization.to_tensor %buffer restrict writable : memref<4xf32> to tensor<4xf32>
  %r = iree_payload.region ins(%tensor : tensor<4xf32>) outs(%tensor : tensor<4xf32>) {
  ^bb0(%source: tensor<4xf32>, %dest: tensor<4xf32>):
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0.0 : f32
    %updated = tensor.insert %zero into %dest[%c0] : tensor<4xf32>
    %old = tensor.extract %source[%c0] : tensor<4xf32>
    %result = tensor.insert %old into %updated[%c1] : tensor<4xf32>
    iree_payload.yield %result : tensor<4xf32>
  } -> tensor<4xf32>
  %resultBuffer = bufferization.to_buffer %r : tensor<4xf32> to memref<4xf32>
  memref.copy %resultBuffer, %buffer : memref<4xf32> to memref<4xf32>
  return
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %scope = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    %buffer_scope = transform.iree.bufferize_payload_boundaries %scope : (!transform.any_op) -> !transform.any_op
    transform.foreach %buffer_scope : !transform.any_op {
    ^bb0(%one_scope: !transform.any_op):
      %payloads = transform.structured.match ops{["iree_payload.region"]} in %one_scope : (!transform.any_op) -> !transform.any_op
    %modules = transform.iree.outline_payload_region %payloads : (!transform.any_op) -> !transform.any_op
      transform.yield
    }
    %empty = transform.structured.match ops{["test.absent"]} in %root : (!transform.any_op) -> !transform.any_op
    %still_empty = transform.iree.bufferize_payload_boundaries %empty : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
