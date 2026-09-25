// RUN: iree-opt --iree-plugin=dispatch_scheduling %s --transform-interpreter --iree-transform-dialect-drop-schedule --iree-llvmcpu-prepare-payload-boundaries | FileCheck %s

// Private results require a final write to the dispatch destination.
// Identical destination views never need such a copy.

// CHECK-LABEL: func.func @private_result(
// CHECK-SAME: %[[SOURCE:[^ ,:]+]]: memref<4xf32>, %[[DEST:[^ ,:]+]]: memref<4xf32>)
// CHECK: %[[PRIVATE:.*]] = memref.alloca()
// CHECK-NEXT: memref.copy %[[DEST]], %[[PRIVATE]]
// CHECK: iree_payload.region ins(%[[SOURCE]] : memref<4xf32>) outs(%[[PRIVATE]] : memref<4xf32>)
// CHECK: llvm.load
// CHECK: llvm.store
// CHECK: iree_payload.yield
// CHECK-NEXT: }
// CHECK-NEXT: memref.copy %[[PRIVATE]], %[[DEST]]
// CHECK-NEXT: return

// CHECK-LABEL: func.func @same_view(
// CHECK-NOT: memref.alloc
// CHECK-NOT: memref.copy
// CHECK: iree_payload.region
// CHECK: llvm.store
// CHECK: iree_payload.yield
// CHECK-NEXT: }
// CHECK-NEXT: return

#target = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {
  target_triple = "aarch64-unknown-unknown-eabi",
  data_layout = "e-m:e-i64:64-i128:128-n32:64-S128",
  cpu = "generic", cpu_features = "+neon"
}>
func.func @private_result(%input: memref<4xf32>, %output: memref<4xf32>) attributes {hal.executable.target = #target} {
  %private = memref.alloca() : memref<4xf32>
  memref.copy %output, %private : memref<4xf32> to memref<4xf32>
  iree_payload.region ins(%input : memref<4xf32>) outs(%private : memref<4xf32>) {
  ^bb0(%src: memref<4xf32>, %dst: memref<4xf32>):
    %c0 = arith.constant 0 : index
    %value = memref.load %src[%c0] : memref<4xf32>
    memref.store %value, %dst[%c0] : memref<4xf32>
    iree_payload.yield
  }
  %result = bufferization.to_tensor %private : memref<4xf32> to tensor<4xf32>
  iree_codegen.store_to_buffer %result, %output : tensor<4xf32> into memref<4xf32>
  return
}
func.func @same_view(%output: memref<4xf32>) attributes {hal.executable.target = #target} {
  iree_payload.region outs(%output : memref<4xf32>) {
  ^bb0(%dst: memref<4xf32>):
    %c0 = arith.constant 0 : index
    %value = arith.constant 1.0 : f32
    memref.store %value, %dst[%c0] : memref<4xf32>
    iree_payload.yield
  }
  %result = bufferization.to_tensor %output : memref<4xf32> to tensor<4xf32>
  iree_codegen.store_to_buffer %result, %output : tensor<4xf32> into memref<4xf32>
  return
}
// Indexed packed-weight inputs have scalar index loads and tensor views
// between the binding adapter and the recorded payload memref arguments.
// CHECK-LABEL: func.func @indexed_views(
// CHECK-NOT: memref.alloc
// CHECK-NOT: memref.copy
// CHECK: %[[ID:.*]] = memref.load %{{.*}}[%{{.*}}] : memref<1xi32>
// CHECK: %[[EXPERT:.*]] = arith.index_cast %[[ID]] : i32 to index
// CHECK: %[[SLICE:.*]] = memref.subview %{{.*}}[%[[EXPERT]], 0, 0] [1, 8, 16] [1, 1, 1]
// CHECK: %[[EXPANDED:.*]] = memref.expand_shape %[[SLICE]]
// CHECK-SAME: into memref<1x8x16xi8, strided<[128, 16, 1], offset: ?>>
// CHECK-NEXT: iree_payload.region ins(%[[EXPANDED]]
// CHECK: llvm.load
// CHECK: llvm.store
// CHECK-NOT: memref.alloc
// CHECK-NOT: memref.copy
// CHECK: return
func.func @indexed_views(%weights: memref<4x8x16xi8>, %ids: memref<1xi32>,
                        %output: memref<1x8x16xi8>) attributes {hal.executable.target = #target} {
  %c0 = arith.constant 0 : index
  %it = iree_codegen.load_from_buffer %ids : memref<1xi32> -> tensor<1xi32>
  %id = tensor.extract %it[%c0] : tensor<1xi32>
  %expert = arith.index_cast %id : i32 to index
  %wt = iree_codegen.load_from_buffer %weights : memref<4x8x16xi8> -> tensor<4x8x16xi8>
  %slice = tensor.extract_slice %wt[%expert, 0, 0] [1, 8, 16] [1, 1, 1]
    : tensor<4x8x16xi8> to tensor<8x16xi8>
  %expanded = tensor.expand_shape %slice [[0, 1], [2]] output_shape [1, 8, 16]
    : tensor<8x16xi8> into tensor<1x8x16xi8>
  %view = bufferization.to_buffer %expanded
    : tensor<1x8x16xi8> to memref<1x8x16xi8, strided<[128, 16, 1], offset: ?>>
  iree_payload.region ins(%view : memref<1x8x16xi8, strided<[128, 16, 1], offset: ?>>)
      outs(%output : memref<1x8x16xi8>) {
  ^bb0(%src: memref<1x8x16xi8, strided<[128, 16, 1], offset: ?>>, %dst: memref<1x8x16xi8>):
    %zero = arith.constant 0 : index
    %value = memref.load %src[%zero, %zero, %zero] : memref<1x8x16xi8, strided<[128, 16, 1], offset: ?>>
    memref.store %value, %dst[%zero, %zero, %zero] : memref<1x8x16xi8>
    iree_payload.yield
  }
  return
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %p = transform.structured.match ops{["iree_payload.region"]} in %root : (!transform.any_op) -> !transform.any_op
    %m = transform.iree.outline_payload_region %p : (!transform.any_op) -> !transform.any_op
    %lowered0 = transform.apply_registered_pass "iree-llvmcpu-prepare-payload-codegen" to %m : (!transform.any_op) -> !transform.any_op
    %lowered1 = transform.apply_registered_pass "expand-strided-metadata" to %lowered0 : (!transform.any_op) -> !transform.any_op
    %lowered2 = transform.apply_registered_pass "lower-affine" to %lowered1 : (!transform.any_op) -> !transform.any_op
    %lowered3 = transform.apply_registered_pass "finalize-memref-to-llvm" to %lowered2 : (!transform.any_op) -> !transform.any_op
    %lowered4 = transform.apply_registered_pass "convert-arith-to-llvm" to %lowered3 : (!transform.any_op) -> !transform.any_op
    %lowered5 = transform.apply_registered_pass "convert-func-to-llvm" to %lowered4 : (!transform.any_op) -> !transform.any_op
    %lowered6 = transform.apply_registered_pass "reconcile-unrealized-casts" to %lowered5 : (!transform.any_op) -> !transform.any_op
    %lowered7 = transform.apply_registered_pass "iree-llvmcpu-verify-payload-codegen" to %lowered6 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
