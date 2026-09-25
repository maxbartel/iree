// RUN: iree-opt --iree-plugin=dispatch_scheduling %s --one-shot-bufferize --canonicalize | FileCheck %s

// Preserve tensor value semantics when yields require destination writes.

// CHECK-LABEL: func.func @unrelated_yield(
// CHECK: iree_payload.region
// CHECK-NEXT: ^bb0(%[[SOURCE:.*]]: memref<4xf32>, %[[DEST:.*]]: memref<4xf32>):
// CHECK-NEXT: memref.copy %[[SOURCE]], %[[DEST]]
// CHECK-NEXT: iree_payload.yield
func.func @unrelated_yield() {
  %a = bufferization.alloc_tensor() : tensor<4xf32>
  %out = bufferization.alloc_tensor() : tensor<4xf32>
  %result = iree_payload.region ins(%a : tensor<4xf32>) outs(%out : tensor<4xf32>) {
  ^bb0(%source: tensor<4xf32>, %dest: tensor<4xf32>):
    iree_payload.yield %source : tensor<4xf32>
  } -> tensor<4xf32>
  return
}

// Both reads must precede either destination write.
// CHECK-LABEL: func.func @swapped_yields(
// CHECK: iree_payload.region
// CHECK-NEXT: ^bb0(%[[FIRST:.*]]: memref<4xf32>, %[[SECOND:.*]]: memref<4xf32>):
// CHECK-NEXT: %[[SECOND_COPY:.*]] = memref.alloc()
// CHECK-NEXT: memref.copy %[[SECOND]], %[[SECOND_COPY]]
// CHECK-NEXT: %[[FIRST_COPY:.*]] = memref.alloc()
// CHECK-NEXT: memref.copy %[[FIRST]], %[[FIRST_COPY]]
// CHECK-NEXT: memref.copy %[[SECOND_COPY]], %[[FIRST]]
// CHECK-NEXT: memref.copy %[[FIRST_COPY]], %[[SECOND]]
// CHECK-NEXT: iree_payload.yield
func.func @swapped_yields() {
  %a = bufferization.alloc_tensor() : tensor<4xf32>
  %b = bufferization.alloc_tensor() : tensor<4xf32>
  %r:2 = iree_payload.region outs(%a, %b : tensor<4xf32>, tensor<4xf32>) {
  ^bb0(%first: tensor<4xf32>, %second: tensor<4xf32>):
    iree_payload.yield %second, %first : tensor<4xf32>, tensor<4xf32>
  } -> tensor<4xf32>, tensor<4xf32>
  return
}

// Inputs have old-tensor semantics; an aliasing destination must be separated
// before executing the body. This is a semantic copy, not a view adapter.
// CHECK-LABEL: func.func @input_output_alias(
// CHECK-SAME: %[[OLD:.*]]: memref<4xf32>)
// CHECK-NEXT: %[[NEW:.*]] = memref.alloc()
// CHECK-NEXT: memref.copy %[[OLD]], %[[NEW]]
// CHECK-NEXT: iree_payload.region ins(%[[OLD]] : memref<4xf32>) outs(%[[NEW]] : memref<4xf32>)
// CHECK-NEXT: ^bb0(%[[SOURCE:.*]]: memref<4xf32>, %[[DEST:.*]]: memref<4xf32>):
// CHECK: memref.store %{{.*}}, %[[DEST]]
// CHECK-NEXT: %[[VALUE:.*]] = memref.load %[[SOURCE]]
// CHECK-NEXT: memref.store %[[VALUE]], %[[DEST]]
// CHECK-NEXT: iree_payload.yield
// CHECK: memref.copy %[[NEW]], %[[OLD]]
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
