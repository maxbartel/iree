// RUN: iree-opt --iree-plugin=dispatch_scheduling %s --transform-interpreter | FileCheck %s

// Combined body/interface bufferization must preserve necessary data movement.
// These are semantic counterexamples to the selected direct-write family.
// No cleanup runs between bufferization and these checks.

// CHECK-LABEL: func.func @unrelated_yield(
// CHECK: iree_payload.region
// CHECK-NEXT: ^bb0(%[[SOURCE:[^ ,:]+]]: memref<4xf32>, %[[DEST:[^ ,:]+]]: memref<4xf32>):
// CHECK: memref.copy %[[SOURCE]], %[[DEST]]
// CHECK: iree_payload.yield

// Snapshot both destinations before writing either one.
// CHECK-LABEL: func.func @swapped_yields(
// CHECK: iree_payload.region
// CHECK-NEXT: ^bb0(%[[FIRST:[^ ,:]+]]: memref<4xf32>, %[[SECOND:[^ ,:]+]]: memref<4xf32>):
// CHECK: %[[SECOND_T:.*]] = bufferization.to_tensor %[[SECOND]]
// CHECK: %[[FIRST_T:.*]] = bufferization.to_tensor %[[FIRST]]
// CHECK: %[[FIRST_B:.*]] = bufferization.to_buffer %[[FIRST_T]]
// CHECK: %[[SECOND_B:.*]] = bufferization.to_buffer %[[SECOND_T]]
// CHECK: %[[COPY_SECOND:.*]] = memref.alloca()
// CHECK-NEXT: memref.copy %[[SECOND_B]], %[[COPY_SECOND]]
// CHECK: %[[COPY_FIRST:.*]] = memref.alloca()
// CHECK-NEXT: memref.copy %[[FIRST_B]], %[[COPY_FIRST]]
// CHECK: memref.copy %[[COPY_SECOND]], %[[FIRST]]
// CHECK-NEXT: memref.copy %[[COPY_FIRST]], %[[SECOND]]
// CHECK-NEXT: iree_payload.yield

// Analysis still covers the enclosing scope: an old-input read must remain
// separate from writes to a tensor initially shared by ins and outs.
// CHECK-LABEL: func.func @input_output_alias(
// CHECK-SAME: %[[OLD:[^ ,:]+]]: memref<4xf32>)
// CHECK: %[[OLD_T:.*]] = bufferization.to_tensor %[[OLD]]
// CHECK: %[[ALLOC:.*]] = memref.alloca()
// CHECK-NEXT: memref.copy %[[OLD]], %[[ALLOC]]
// CHECK: %[[SNAPSHOT:.*]] = bufferization.to_tensor %[[ALLOC]]
// CHECK: %[[NEW:.*]] = bufferization.to_buffer %[[SNAPSHOT]]
// CHECK: iree_payload.region ins(%[[OLD]] : memref<4xf32>) outs(%[[NEW]] : memref<4xf32>)
// CHECK: memref.store
// CHECK: %[[READ:.*]] = memref.load
// CHECK-NEXT: memref.store %[[READ]]
// CHECK: iree_payload.yield
// CHECK: memref.copy

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
// A loop that selects a different descriptor needs a destination write. Merely
// starting the loop with the output buffer is insufficient to omit that copy.
// CHECK-LABEL: func.func @changed_descriptor(
// CHECK: iree_payload.region
// CHECK-NEXT: ^bb0(%{{.*}}: memref<4xf32>, %{{.*}}: index, %[[DEST:[^ ,:]+]]: memref<4xf32>):
// CHECK: %[[SELECTED:.*]] = scf.for
// CHECK: scf.yield
// CHECK: memref.copy %[[SELECTED]], %[[DEST]]
// CHECK-NEXT: iree_payload.yield
func.func @changed_descriptor(%a: memref<4xf32>, %b: memref<4xf32>, %n: index) {
  %input = bufferization.to_tensor %a restrict : memref<4xf32> to tensor<4xf32>
  %init = bufferization.to_tensor %b restrict writable : memref<4xf32> to tensor<4xf32>
  %r = iree_payload.region ins(%input, %n : tensor<4xf32>, index) outs(%init : tensor<4xf32>) {
  ^bb0(%source: tensor<4xf32>, %count: index, %dest: tensor<4xf32>):
    %in_buf = bufferization.to_buffer %source : tensor<4xf32> to memref<4xf32>
    %out_buf = bufferization.to_buffer %dest : tensor<4xf32> to memref<4xf32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %selected = scf.for %i = %c0 to %count step %c1 iter_args(%buffer = %out_buf) -> memref<4xf32> {
      scf.yield %in_buf : memref<4xf32>
    }
    %result = bufferization.to_tensor %selected restrict : memref<4xf32> to tensor<4xf32>
    iree_payload.yield %result : tensor<4xf32>
  } -> tensor<4xf32>
  return
}

// A conditional that can yield a different descriptor must still copy.
// CHECK-LABEL: func.func @conditional_input(
// CHECK: iree_payload.region
// CHECK-NEXT: ^bb0(%{{.*}}: memref<4xf32>, %{{.*}}: i1, %[[DEST:[^ ,:]+]]: memref<4xf32>):
// CHECK: %[[SELECTED:.*]] = scf.if
// CHECK: memref.copy %[[SELECTED]], %[[DEST]]
// CHECK-NEXT: iree_payload.yield
func.func @conditional_input(%src: memref<4xf32>, %dst: memref<4xf32>, %cond: i1) {
  %input = bufferization.to_tensor %src restrict : memref<4xf32> to tensor<4xf32>
  %init = bufferization.to_tensor %dst restrict writable : memref<4xf32> to tensor<4xf32>
  %r = iree_payload.region ins(%input, %cond : tensor<4xf32>, i1) outs(%init : tensor<4xf32>) {
  ^bb0(%source: tensor<4xf32>, %c: i1, %dest: tensor<4xf32>):
    %a = bufferization.to_buffer %source : tensor<4xf32> to memref<4xf32>
    %b = bufferization.to_buffer %dest : tensor<4xf32> to memref<4xf32>
    %selected = scf.if %c -> memref<4xf32> {
      scf.yield %a : memref<4xf32>
    } else {
      scf.yield %b : memref<4xf32>
    }
    %result = bufferization.to_tensor %selected restrict : memref<4xf32> to tensor<4xf32>
    iree_payload.yield %result : tensor<4xf32>
  } -> tensor<4xf32>
  return
}

// Equal base buffers and subview types do not imply equal descriptors. Moving
// a strided slice to a distinct offset must still write the destination slice.
// CHECK-LABEL: func.func @moved_slice(
// CHECK: %[[SOURCE:.*]] = memref.subview %{{.*}}[0] [4] [2]
// CHECK: memref.copy %[[SOURCE]], %[[SNAPSHOT:[^ :]+]]
// CHECK: %[[DEST:.*]] = memref.subview %{{.*}}[8] [4] [2]
// CHECK: memref.copy %[[SNAPSHOT]], %[[DEST]]
// CHECK: iree_payload.yield
func.func @moved_slice(%dst: memref<16xf32>) {
  %init = bufferization.to_tensor %dst restrict writable : memref<16xf32> to tensor<16xf32>
  %r = iree_payload.region outs(%init : tensor<16xf32>) {
  ^bb0(%out: tensor<16xf32>):
    %source = tensor.extract_slice %out[0] [4] [2] : tensor<16xf32> to tensor<4xf32>
    %result = tensor.insert_slice %source into %out[8] [4] [2] : tensor<4xf32> into tensor<16xf32>
    iree_payload.yield %result : tensor<16xf32>
  } -> tensor<16xf32>
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %scope = transform.structured.match ops{["func.func"]} in %root
      : (!transform.any_op) -> !transform.any_op
    %buffers = transform.iree.bufferize_payload_boundaries %scope <{bufferize_body}>
      : (!transform.any_op) -> !transform.any_op
    transform.foreach %buffers : !transform.any_op {
    ^bb0(%one: !transform.any_op):
      %payload = transform.structured.match ops{["iree_payload.region"]} in %one
        : (!transform.any_op) -> !transform.any_op
      transform.yield
    }
    transform.yield
  }
}
