// RUN: iree-opt --iree-plugin=dispatch_scheduling %s --transform-interpreter --canonicalize -o %t
// RUN: FileCheck %s --check-prefix=BOUNDARY < %t
// RUN: iree-opt --iree-plugin=dispatch_scheduling %t --pass-pipeline='builtin.module(func.func(iree_payload.region(builtin.module(func.func(iree-codegen-iree-comprehensive-bufferize,canonicalize)))))' | FileCheck %s --check-prefix=FULL --implicit-check-not=memref.copy --implicit-check-not=memref.alloc --implicit-check-not=bufferization.

// The interface retains exact destination descriptors, while the outlined
// function remains in tensor form for ordinary scheduling passes. Full body
// bufferization writes directly to the same destination without a final copy.
// BOUNDARY-LABEL: func.func @compute(
// BOUNDARY-SAME: %[[DEST:[^ ,:]+]]: memref<2x3x8x8xf32>)
// BOUNDARY: iree_payload.region
// BOUNDARY-SAME: outs(%[[DEST]] : memref<2x3x8x8xf32>)
// BOUNDARY: func.func @entry(
// BOUNDARY: bufferization.to_tensor
// BOUNDARY: linalg.mmt4d
// BOUNDARY-SAME: tensor<2x3x8x8xf32>
// FULL-LABEL: func.func @compute(
// FULL: func.func @entry(%[[A:[^ ,:]+]]: memref<2x4x8x1xf32>, %[[B:[^ ,:]+]]: memref<3x4x8x1xf32>, %[[O:[^ ,:]+]]: memref<2x3x8x8xf32>)
// FULL: linalg.fill
// FULL-SAME: outs(%[[O]] : memref<2x3x8x8xf32>)
// FULL-NEXT: linalg.mmt4d ins(%[[A]], %[[B]]
// FULL-SAME: outs(%[[O]] : memref<2x3x8x8xf32>)
// FULL-NEXT: return

// BOUNDARY-LABEL: func.func @dynamic_view(
// BOUNDARY: func.func @entry(%{{.*}}: memref<?xf32, strided<[?], offset: ?>>, %{{.*}}: f32, %{{.*}}: memref<?xf32, strided<[?], offset: ?>>)
// BOUNDARY: linalg.generic
// BOUNDARY-SAME: tensor<?xf32>
// FULL-LABEL: func.func @dynamic_view(
// FULL: func.func @entry(%[[IN:[^ ,:]+]]: memref<?xf32, strided<[?], offset: ?>>, %[[S:[^ ,:]+]]: f32, %[[OUT:[^ ,:]+]]: memref<?xf32, strided<[?], offset: ?>>)
// FULL-NEXT: linalg.generic
// FULL-SAME: ins(%[[IN]] : memref<?xf32, strided<[?], offset: ?>>) outs(%[[OUT]] : memref<?xf32, strided<[?], offset: ?>>)
// FULL: arith.addf %{{.*}}, %[[S]]
// FULL: linalg.yield
// FULL-NEXT: }
// FULL-NEXT: return

// A scope without payloads retains tensor operations, and the returned scope
// handle works for every matched function. An empty handle is harmless.
// BOUNDARY-LABEL: func.func @no_payload(
// BOUNDARY: tensor.insert
// FULL-LABEL: func.func @no_payload(
// FULL: tensor.insert

func.func @compute(%a: memref<2x4x8x1xf32>, %b: memref<3x4x8x1xf32>, %o: memref<2x3x8x8xf32>) {
  %ta = bufferization.to_tensor %a restrict : memref<2x4x8x1xf32> to tensor<2x4x8x1xf32>
  %tb = bufferization.to_tensor %b restrict : memref<3x4x8x1xf32> to tensor<3x4x8x1xf32>
  %to = bufferization.to_tensor %o restrict writable : memref<2x3x8x8xf32> to tensor<2x3x8x8xf32>
  %r = iree_payload.region ins(%ta, %tb : tensor<2x4x8x1xf32>, tensor<3x4x8x1xf32>) outs(%to : tensor<2x3x8x8xf32>) {
  ^bb0(%aa: tensor<2x4x8x1xf32>, %bb: tensor<3x4x8x1xf32>, %oo: tensor<2x3x8x8xf32>):
    %zero = arith.constant 0.0 : f32
    %f = linalg.fill ins(%zero : f32) outs(%oo : tensor<2x3x8x8xf32>) -> tensor<2x3x8x8xf32>
    %mm = linalg.mmt4d ins(%aa, %bb : tensor<2x4x8x1xf32>, tensor<3x4x8x1xf32>) outs(%f : tensor<2x3x8x8xf32>) -> tensor<2x3x8x8xf32>
    iree_payload.yield %mm : tensor<2x3x8x8xf32>
  } -> tensor<2x3x8x8xf32>
  %rb = bufferization.to_buffer %r : tensor<2x3x8x8xf32> to memref<2x3x8x8xf32>
  memref.copy %rb, %o : memref<2x3x8x8xf32> to memref<2x3x8x8xf32>
  return
}


func.func @dynamic_view(%source: memref<?xf32, strided<[?], offset: ?>>,
                        %out: memref<?xf32, strided<[?], offset: ?>>,
                        %scale: f32) {
  %input = bufferization.to_tensor %source restrict : memref<?xf32, strided<[?], offset: ?>> to tensor<?xf32>
  %init = bufferization.to_tensor %out restrict writable : memref<?xf32, strided<[?], offset: ?>> to tensor<?xf32>
  %r = iree_payload.region ins(%input, %scale : tensor<?xf32>, f32) outs(%init : tensor<?xf32>) {
  ^bb0(%i: tensor<?xf32>, %s: f32, %o: tensor<?xf32>):
    %v = linalg.generic {indexing_maps = [affine_map<(i)->(i)>, affine_map<(i)->(i)>], iterator_types = ["parallel"]} ins(%i : tensor<?xf32>) outs(%o : tensor<?xf32>) {
    ^bb0(%x: f32, %y: f32):
      %sum = arith.addf %x, %s : f32
      linalg.yield %sum : f32
    } -> tensor<?xf32>
    iree_payload.yield %v : tensor<?xf32>
  } -> tensor<?xf32>
  %buffer = bufferization.to_buffer %r : tensor<?xf32> to memref<?xf32, strided<[?], offset: ?>>
  memref.copy %buffer, %out : memref<?xf32, strided<[?], offset: ?>> to memref<?xf32, strided<[?], offset: ?>>
  return
}

func.func @no_payload(%value: tensor<4xf32>) -> tensor<4xf32> {
  %c0 = arith.constant 0 : index
  %zero = arith.constant 0.0 : f32
  %r = tensor.insert %zero into %value[%c0] : tensor<4xf32>
  return %r : tensor<4xf32>
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
