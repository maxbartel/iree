// RUN: iree-opt --iree-plugin=dispatch_scheduling %s --transform-interpreter --iree-transform-dialect-drop-schedule | FileCheck %s

// Different tensor SSA values may still load overlapping binding storage.
// Preserve the old source before a payload writes an overlapping destination.
// No CSE or copy-removing cleanup runs before these checks.

// CHECK-LABEL: func.func @identical_views(
// CHECK: %[[SOURCE:.*]] = iree_tensor_ext.dispatch.tensor.load
// CHECK: %[[SOURCE_B:.*]] = bufferization.to_buffer %[[SOURCE]]
// CHECK: %[[DEST:.*]] = iree_tensor_ext.dispatch.tensor.load
// CHECK: %[[DEST_B:.*]] = bufferization.to_buffer %[[DEST]]
// CHECK: %[[ALLOC:.*]] = memref.alloca()
// CHECK-NEXT: memref.copy %[[DEST_B]], %[[ALLOC]]
// CHECK: %[[NEW_T:.*]] = bufferization.to_tensor %[[ALLOC]]
// CHECK: %[[NEW_B:.*]] = bufferization.to_buffer %[[NEW_T]]
// CHECK: iree_payload.region ins(%[[SOURCE_B]] :
// CHECK-SAME: outs(%[[NEW_B]] : memref<4xf32>)
// CHECK: memref.store
// CHECK: memref.load
// CHECK: iree_tensor_ext.dispatch.tensor.store
func.func @identical_views(%binding: !iree_tensor_ext.dispatch.tensor<readwrite:tensor<8xf32>>) {
  %source = iree_tensor_ext.dispatch.tensor.load %binding, offsets = [0], sizes = [4], strides = [1] : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<8xf32>> -> tensor<4xf32>
  %dest = iree_tensor_ext.dispatch.tensor.load %binding, offsets = [0], sizes = [4], strides = [1] : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<8xf32>> -> tensor<4xf32>
  %result = iree_payload.region ins(%source : tensor<4xf32>) outs(%dest : tensor<4xf32>) {
  ^bb0(%old: tensor<4xf32>, %init: tensor<4xf32>):
    %c0 = arith.constant 0 : index
    %write = arith.constant 0 : index
    %next = arith.constant 2 : index
    %zero = arith.constant 0.0 : f32
    %updated = tensor.insert %zero into %init[%write] : tensor<4xf32>
    %value = tensor.extract %old[%c0] : tensor<4xf32>
    %final = tensor.insert %value into %updated[%next] : tensor<4xf32>
    iree_payload.yield %final : tensor<4xf32>
  } -> tensor<4xf32>
  iree_tensor_ext.dispatch.tensor.store %result, %binding, offsets = [0], sizes = [4], strides = [1] : tensor<4xf32> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<8xf32>>
  return
}

// Shape metadata must not hide the shared binding from alias analysis.
// CHECK-LABEL: func.func @shape_tied_views(
// CHECK: %[[SOURCE:.*]] = iree_tensor_ext.dispatch.tensor.load
// CHECK: %[[SOURCE_B:.*]] = bufferization.to_buffer %[[SOURCE]]
// CHECK: %[[DEST:.*]] = iree_tensor_ext.dispatch.tensor.load
// CHECK: %[[DEST_B:.*]] = bufferization.to_buffer %[[DEST]]
// CHECK: %[[ALLOC:.*]] = memref.alloca()
// CHECK-NEXT: memref.copy %[[DEST_B]], %[[ALLOC]]
// CHECK: %[[NEW_T:.*]] = bufferization.to_tensor %[[ALLOC]]
// CHECK: %[[NEW_B:.*]] = bufferization.to_buffer %[[NEW_T]]
// CHECK: iree_payload.region ins(%[[SOURCE_B]] :
// CHECK-SAME: outs(%[[NEW_B]] : memref<4xf32>)
// CHECK: memref.store
// CHECK: memref.load
// CHECK: iree_tensor_ext.dispatch.tensor.store
func.func @shape_tied_views(%binding: !iree_tensor_ext.dispatch.tensor<readwrite:tensor<8xf32>>) {
  %source = iree_tensor_ext.dispatch.tensor.load %binding, offsets = [0], sizes = [4], strides = [1] : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<8xf32>> -> tensor<4xf32>
  %tied = flow.dispatch.tie_shape %binding : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<8xf32>>{}
  %dest = iree_tensor_ext.dispatch.tensor.load %tied, offsets = [0], sizes = [4], strides = [1] : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<8xf32>> -> tensor<4xf32>
  %result = iree_payload.region ins(%source : tensor<4xf32>) outs(%dest : tensor<4xf32>) {
  ^bb0(%old: tensor<4xf32>, %init: tensor<4xf32>):
    %c0 = arith.constant 0 : index
    %write = arith.constant 0 : index
    %next = arith.constant 2 : index
    %zero = arith.constant 0.0 : f32
    %updated = tensor.insert %zero into %init[%write] : tensor<4xf32>
    %value = tensor.extract %old[%c0] : tensor<4xf32>
    %final = tensor.insert %value into %updated[%next] : tensor<4xf32>
    iree_payload.yield %final : tensor<4xf32>
  } -> tensor<4xf32>
  iree_tensor_ext.dispatch.tensor.store %result, %binding, offsets = [0], sizes = [4], strides = [1] : tensor<4xf32> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<8xf32>>
  return
}

// CHECK-LABEL: func.func @overlapping_views(
// CHECK: %[[SOURCE:.*]] = iree_tensor_ext.dispatch.tensor.load
// CHECK: %[[SOURCE_B:.*]] = bufferization.to_buffer %[[SOURCE]]
// CHECK: %[[DEST:.*]] = iree_tensor_ext.dispatch.tensor.load
// CHECK: %[[DEST_B:.*]] = bufferization.to_buffer %[[DEST]]
// CHECK: %[[ALLOC:.*]] = memref.alloca()
// CHECK-NEXT: memref.copy %[[DEST_B]], %[[ALLOC]]
// CHECK: %[[NEW_T:.*]] = bufferization.to_tensor %[[ALLOC]]
// CHECK: %[[NEW_B:.*]] = bufferization.to_buffer %[[NEW_T]]
// CHECK: iree_payload.region ins(%[[SOURCE_B]] :
// CHECK-SAME: outs(%[[NEW_B]] : memref<4xf32>)
// CHECK: memref.store
// CHECK: memref.load
// CHECK: iree_tensor_ext.dispatch.tensor.store
func.func @overlapping_views(%binding: !iree_tensor_ext.dispatch.tensor<readwrite:tensor<8xf32>>) {
  %source = iree_tensor_ext.dispatch.tensor.load %binding, offsets = [1], sizes = [4], strides = [1] : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<8xf32>> -> tensor<4xf32>
  %dest = iree_tensor_ext.dispatch.tensor.load %binding, offsets = [0], sizes = [4], strides = [1] : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<8xf32>> -> tensor<4xf32>
  %result = iree_payload.region ins(%source : tensor<4xf32>) outs(%dest : tensor<4xf32>) {
  ^bb0(%old: tensor<4xf32>, %init: tensor<4xf32>):
    %c0 = arith.constant 0 : index
    %write = arith.constant 1 : index
    %next = arith.constant 2 : index
    %zero = arith.constant 0.0 : f32
    %updated = tensor.insert %zero into %init[%write] : tensor<4xf32>
    %value = tensor.extract %old[%c0] : tensor<4xf32>
    %final = tensor.insert %value into %updated[%next] : tensor<4xf32>
    iree_payload.yield %final : tensor<4xf32>
  } -> tensor<4xf32>
  iree_tensor_ext.dispatch.tensor.store %result, %binding, offsets = [0], sizes = [4], strides = [1] : tensor<4xf32> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<8xf32>>
  return
}

// Distinct logical bindings do not need a preservation allocation.
// CHECK-LABEL: func.func @independent_bindings(
// CHECK-NOT: memref.alloc
// CHECK-NOT: memref.copy
// CHECK-NOT: bufferization.alloc_tensor
// CHECK: iree_tensor_ext.dispatch.tensor.store
func.func @independent_bindings(%source_binding: !iree_tensor_ext.dispatch.tensor<readonly:tensor<8xf32>>, %binding: !iree_tensor_ext.dispatch.tensor<readwrite:tensor<8xf32>>) {
  %source = iree_tensor_ext.dispatch.tensor.load %source_binding, offsets = [0], sizes = [4], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8xf32>> -> tensor<4xf32>
  %dest = iree_tensor_ext.dispatch.tensor.load %binding, offsets = [0], sizes = [4], strides = [1] : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<8xf32>> -> tensor<4xf32>
  %result = iree_payload.region ins(%source : tensor<4xf32>) outs(%dest : tensor<4xf32>) {
  ^bb0(%old: tensor<4xf32>, %init: tensor<4xf32>):
    %c0 = arith.constant 0 : index
    %write = arith.constant 0 : index
    %next = arith.constant 2 : index
    %zero = arith.constant 0.0 : f32
    %updated = tensor.insert %zero into %init[%write] : tensor<4xf32>
    %value = tensor.extract %old[%c0] : tensor<4xf32>
    %final = tensor.insert %value into %updated[%next] : tensor<4xf32>
    iree_payload.yield %final : tensor<4xf32>
  } -> tensor<4xf32>
  iree_tensor_ext.dispatch.tensor.store %result, %binding, offsets = [0], sizes = [4], strides = [1] : tensor<4xf32> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<8xf32>>
  return
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
