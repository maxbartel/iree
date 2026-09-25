// RUN: iree-opt --iree-plugin=dispatch_scheduling %s --transform-interpreter --iree-transform-dialect-drop-schedule | FileCheck %s --implicit-check-not=memref.copy --implicit-check-not=memref.alloc

// Preserve strides derived from a dispatch argument and its selected slice.
// The binding offset remains unknown. This test inspects raw bufferization.

// CHECK-LABEL: func.func @static_slice(
// CHECK: iree_payload.region outs(%{{.*}} : memref<2x8xf32, strided<[64, 3], offset: ?>>)
// CHECK: memref.load
// CHECK: memref.store

func.func @static_slice(%source: !iree_tensor_ext.dispatch.tensor<readwrite:tensor<8x32xf32>>) -> tensor<2x8xf32> {
  %view = iree_tensor_ext.dispatch.tensor.load %source,
    offsets = [2, 4], sizes = [2, 8], strides = [2, 3]
    : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<8x32xf32>> -> tensor<2x8xf32>
  %r = iree_payload.region outs(%view : tensor<2x8xf32>) {
  ^bb0(%out: tensor<2x8xf32>):
    %c0 = arith.constant 0 : index
    %one = arith.constant 1.0 : f32
    %old = tensor.extract %out[%c0, %c0] : tensor<2x8xf32>
    %sum = arith.addf %old, %one : f32
    %next = tensor.insert %sum into %out[%c0, %c0] : tensor<2x8xf32>
    iree_payload.yield %next : tensor<2x8xf32>
  } -> tensor<2x8xf32>
  return %r : tensor<2x8xf32>
}

// CHECK-LABEL: func.func @dynamic_dense(
// CHECK: iree_payload.region outs(%{{.*}} : memref<?x?x8xf32, strided<[?, 8, 1], offset: ?>>)
// CHECK: memref.load
// CHECK: memref.store

func.func @dynamic_dense(%source: !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?x8xf32>>, %m: index, %n: index) -> tensor<?x?x8xf32> {
  %shaped = flow.dispatch.tie_shape %source : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?x8xf32>>{%m, %n}
  %view = iree_tensor_ext.dispatch.tensor.load %shaped,
    offsets = [0, 0, 0], sizes = [%m, %n, 8], strides = [1, 1, 1]
    : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?x8xf32>>{%m, %n} -> tensor<?x?x8xf32>
  %r = iree_payload.region outs(%view : tensor<?x?x8xf32>) {
  ^bb0(%out: tensor<?x?x8xf32>):
    %c0 = arith.constant 0 : index
    %one = arith.constant 1.0 : f32
    %old = tensor.extract %out[%c0, %c0, %c0] : tensor<?x?x8xf32>
    %sum = arith.addf %old, %one : f32
    %next = tensor.insert %sum into %out[%c0, %c0, %c0] : tensor<?x?x8xf32>
    iree_payload.yield %next : tensor<?x?x8xf32>
  } -> tensor<?x?x8xf32>
  return %r : tensor<?x?x8xf32>
}

// CHECK-LABEL: func.func @rank_reduced(
// CHECK: iree_payload.region outs(%{{.*}} : memref<8x4xf32, strided<[16, 2], offset: ?>>)
// CHECK: memref.load
// CHECK: memref.store

func.func @rank_reduced(%source: !iree_tensor_ext.dispatch.tensor<readwrite:tensor<4x8x16xf32>>) -> tensor<8x4xf32> {
  %view = iree_tensor_ext.dispatch.tensor.load %source,
    offsets = [2, 0, 1], sizes = [1, 8, 4], strides = [1, 1, 2]
    : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<4x8x16xf32>> -> tensor<8x4xf32>
  %r = iree_payload.region outs(%view : tensor<8x4xf32>) {
  ^bb0(%out: tensor<8x4xf32>):
    %c0 = arith.constant 0 : index
    %one = arith.constant 1.0 : f32
    %old = tensor.extract %out[%c0, %c0] : tensor<8x4xf32>
    %sum = arith.addf %old, %one : f32
    %next = tensor.insert %sum into %out[%c0, %c0] : tensor<8x4xf32>
    iree_payload.yield %next : tensor<8x4xf32>
  } -> tensor<8x4xf32>
  return %r : tensor<8x4xf32>
}

// CHECK-LABEL: func.func @dynamic_stride(
// CHECK: iree_payload.region outs(%{{.*}} : memref<4x8xf32, strided<[?, 2], offset: ?>>)
// CHECK: memref.load
// CHECK: memref.store

func.func @dynamic_stride(%source: !iree_tensor_ext.dispatch.tensor<readwrite:tensor<16x32xf32>>, %offset: index, %stride: index) -> tensor<4x8xf32> {
  %view = iree_tensor_ext.dispatch.tensor.load %source,
    offsets = [%offset, 1], sizes = [4, 8], strides = [%stride, 2]
    : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<16x32xf32>> -> tensor<4x8xf32>
  %r = iree_payload.region outs(%view : tensor<4x8xf32>) {
  ^bb0(%out: tensor<4x8xf32>):
    %c0 = arith.constant 0 : index
    %one = arith.constant 1.0 : f32
    %old = tensor.extract %out[%c0, %c0] : tensor<4x8xf32>
    %sum = arith.addf %old, %one : f32
    %next = tensor.insert %sum into %out[%c0, %c0] : tensor<4x8xf32>
    iree_payload.yield %next : tensor<4x8xf32>
  } -> tensor<4x8xf32>
  return %r : tensor<4x8xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %funcs = transform.structured.match ops{["func.func"]} in %root
      : (!transform.any_op) -> !transform.any_op
    %buffers = transform.iree.bufferize_payload_boundaries %funcs <{bufferize_body}>
      : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
