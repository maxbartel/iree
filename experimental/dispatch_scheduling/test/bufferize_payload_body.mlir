// RUN: iree-opt --iree-plugin=dispatch_scheduling %s --transform-interpreter | FileCheck %s --implicit-check-not=memref.copy --implicit-check-not=memref.alloc --implicit-check-not=linalg.copy

// Inspect the immediate bufferization result. There is deliberately no cleanup
// pass that could erase copies. Nested loops forward the same destination view,
// including when the dynamic outer loop executes zero times. Input and output
// descriptors have different non-unit strides and dynamic offsets.
// CHECK-LABEL: func.func @direct_write(
// CHECK: iree_payload.region
// CHECK-NEXT: ^bb0(%[[IN:[^ ,:]+]]: memref<?x8xf32, strided<[?, 2], offset: ?>>, %[[N:[^ ,:]+]]: index, %[[OUT:[^ ,:]+]]: memref<?x8xf32, strided<[?, 3], offset: ?>>):
// CHECK: %[[OUT_T:.*]] = bufferization.to_tensor %[[OUT]]
// CHECK: %[[IN_T:.*]] = bufferization.to_tensor %[[IN]]
// CHECK: %[[OUT_B:.*]] = bufferization.to_buffer %[[OUT_T]]
// CHECK: %[[IN_B:.*]] = bufferization.to_buffer %[[IN_T]]
// CHECK: scf.for
// CHECK-SAME: iter_args(%[[ROW:[^ ,:]+]] = %[[OUT_B]])
// CHECK: %[[ROW_T:.*]] = bufferization.to_tensor %[[ROW]]
// CHECK: %[[ROW_B:.*]] = bufferization.to_buffer %[[ROW_T]]
// CHECK: scf.for
// CHECK-SAME: iter_args(%[[DST:[^ ,:]+]] = %[[ROW_B]])
// CHECK: %[[DST_T:.*]] = bufferization.to_tensor %[[DST]]
// CHECK: %[[DST_B:.*]] = bufferization.to_buffer %[[DST_T]]
// CHECK: vector.transfer_read %[[IN_B]]
// CHECK: arith.addf
// CHECK: vector.transfer_write %{{.*}}, %[[DST_B]]
// CHECK: iree_payload.yield

// A function without a payload still has tensor semantics.
// CHECK-LABEL: func.func @outside(
// CHECK: tensor.insert

func.func @direct_write(%a: memref<?x8xf32, strided<[?, 2], offset: ?>>,
                        %b: memref<?x8xf32, strided<[?, 3], offset: ?>>,
                        %rows: index) {
  %input = bufferization.to_tensor %a restrict : memref<?x8xf32, strided<[?, 2], offset: ?>> to tensor<?x8xf32>
  %init = bufferization.to_tensor %b restrict writable : memref<?x8xf32, strided<[?, 3], offset: ?>> to tensor<?x8xf32>
  %result = iree_payload.region ins(%input, %rows : tensor<?x8xf32>, index) outs(%init : tensor<?x8xf32>) {
  ^bb0(%in: tensor<?x8xf32>, %n: index, %out: tensor<?x8xf32>):
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %zero = arith.constant 0.0 : f32
    %one = arith.constant dense<1.0> : vector<1xf32>
    %r = scf.for %i = %c0 to %n step %c1 iter_args(%current = %out) -> tensor<?x8xf32> {
      %row = scf.for %j = %c0 to %c8 step %c1 iter_args(%dst = %current) -> tensor<?x8xf32> {
        %v = vector.transfer_read %in[%i, %j], %zero {in_bounds = [true]} : tensor<?x8xf32>, vector<1xf32>
        %sum = arith.addf %v, %one : vector<1xf32>
        %next = vector.transfer_write %sum, %dst[%i, %j] {in_bounds = [true]} : vector<1xf32>, tensor<?x8xf32>
        scf.yield %next : tensor<?x8xf32>
      }
      scf.yield %row : tensor<?x8xf32>
    }
    iree_payload.yield %r : tensor<?x8xf32>
  } -> tensor<?x8xf32>
  return
}

func.func @outside(%value: tensor<4xf32>) -> tensor<4xf32> {
  %c0 = arith.constant 0 : index
  %zero = arith.constant 0.0 : f32
  %r = tensor.insert %zero into %value[%c0] : tensor<4xf32>
  return %r : tensor<4xf32>
}

// Both branches forward exactly the final destination, including when one
// branch contains a dynamic loop. No closing copy may be constructed.
// CHECK-LABEL: func.func @conditional_destination(
// CHECK: iree_payload.region
// CHECK: scf.if
// CHECK: scf.for
// CHECK: vector.transfer_write
// CHECK: iree_payload.yield
func.func @conditional_destination(%dst: memref<8xf32>, %cond: i1, %n: index) {
  %init = bufferization.to_tensor %dst restrict writable : memref<8xf32> to tensor<8xf32>
  %r = iree_payload.region ins(%cond, %n : i1, index) outs(%init : tensor<8xf32>) {
  ^bb0(%c: i1, %count: index, %out: tensor<8xf32>):
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %v = arith.constant dense<1.0> : vector<8xf32>
    %selected = scf.if %c -> tensor<8xf32> {
      %loop = scf.for %i = %c0 to %count step %c1 iter_args(%current = %out) -> tensor<8xf32> {
        %next = vector.transfer_write %v, %current[%c0] {in_bounds = [true]} : vector<8xf32>, tensor<8xf32>
        scf.yield %next : tensor<8xf32>
      }
      scf.yield %loop : tensor<8xf32>
    } else {
      scf.yield %out : tensor<8xf32>
    }
    iree_payload.yield %selected : tensor<8xf32>
  } -> tensor<8xf32>
  return
}

// Masked writes to a dynamic, rank-reduced destination must not create a
// copy between two identical subviews when reinserting the tensor slice.
// CHECK-LABEL: func.func @masked_dynamic_slice(
// CHECK: iree_payload.region
// CHECK: %[[SLICE:.*]] = memref.subview
// CHECK: %[[MASK:.*]] = vector.create_mask
// CHECK: vector.transfer_write %{{.*}}, %[[SLICE]][%{{.*}}], %[[MASK]]
// CHECK: iree_payload.yield
func.func @masked_dynamic_slice(%dst: memref<1x1x?xf32, strided<[?, ?, 1], offset: ?>>) {
  %init = bufferization.to_tensor %dst restrict writable
    : memref<1x1x?xf32, strided<[?, ?, 1], offset: ?>> to tensor<1x1x?xf32>
  %r = iree_payload.region outs(%init : tensor<1x1x?xf32>) {
  ^bb0(%out: tensor<1x1x?xf32>):
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %n = tensor.dim %out, %c2 : tensor<1x1x?xf32>
    %slice = tensor.extract_slice %out[0, 0, 0] [1, 1, %n] [1, 1, 1]
      : tensor<1x1x?xf32> to tensor<?xf32>
    %mask = vector.create_mask %n : vector<32xi1>
    %zeros = arith.constant dense<0.0> : vector<32xf32>
    %written = vector.transfer_write %zeros, %slice[%c0], %mask {in_bounds = [true]}
      : vector<32xf32>, tensor<?xf32>
    %inserted = tensor.insert_slice %written into %out[0, 0, 0] [1, 1, %n] [1, 1, 1]
      : tensor<?xf32> into tensor<1x1x?xf32>
    iree_payload.yield %inserted : tensor<1x1x?xf32>
  } -> tensor<1x1x?xf32>
  return
}

// A captured initializer can become unused after tensor/vector scheduling.
// Its alias with the destination does not require reading or copying storage.
// CHECK-LABEL: func.func @unused_input_aliases_destination(
// CHECK: iree_payload.region ins(%arg0 : memref<8xf32>) outs(%arg0 : memref<8xf32>)
// CHECK: vector.transfer_write
// CHECK: iree_payload.yield
func.func @unused_input_aliases_destination(%dst: memref<8xf32>) {
  %init = bufferization.to_tensor %dst restrict writable
    : memref<8xf32> to tensor<8xf32>
  %r = iree_payload.region ins(%init : tensor<8xf32>) outs(%init : tensor<8xf32>) {
  ^bb0(%unused: tensor<8xf32>, %out: tensor<8xf32>):
    %c0 = arith.constant 0 : index
    %zeros = arith.constant dense<0.0> : vector<8xf32>
    %written = vector.transfer_write %zeros, %out[%c0] {in_bounds = [true]}
      : vector<8xf32>, tensor<8xf32>
    iree_payload.yield %written : tensor<8xf32>
  } -> tensor<8xf32>
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
