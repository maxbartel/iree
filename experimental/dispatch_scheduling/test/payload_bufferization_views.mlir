// RUN: iree-opt --iree-plugin=dispatch_scheduling %s --one-shot-bufferize | FileCheck %s --implicit-check-not=memref.copy --implicit-check-not=memref.alloc --implicit-check-not=bufferization.
// RUN: iree-opt --iree-plugin=dispatch_scheduling %s --one-shot-bufferize --canonicalize | FileCheck %s --implicit-check-not=memref.copy --implicit-check-not=memref.alloc --implicit-check-not=bufferization.

// Tensor/buffer boundary adapters must preserve exact views without copies.

// CHECK-LABEL: func.func @inplace(
// CHECK: iree_payload.region ins(%{{.*}}, %{{.*}} : memref<4x4xf32>, memref<4x4xf32>) outs(%{{.*}} : memref<4x4xf32>)
// CHECK-NEXT: ^bb0(%[[A:.*]]: memref<4x4xf32>, %[[B:.*]]: memref<4x4xf32>, %[[C:.*]]: memref<4x4xf32>):
// CHECK-NEXT: linalg.matmul ins(%[[A]], %[[B]] : memref<4x4xf32>, memref<4x4xf32>) outs(%[[C]] : memref<4x4xf32>)
// CHECK-NEXT: iree_payload.yield
func.func @inplace(%ab: memref<4x4xf32>, %bb: memref<4x4xf32>, %cb: memref<4x4xf32>) {
  %a = bufferization.to_tensor %ab restrict : memref<4x4xf32> to tensor<4x4xf32>
  %b = bufferization.to_tensor %bb restrict : memref<4x4xf32> to tensor<4x4xf32>
  %out = bufferization.to_tensor %cb restrict writable : memref<4x4xf32> to tensor<4x4xf32>
  %result = iree_payload.region ins(%a, %b : tensor<4x4xf32>, tensor<4x4xf32>) outs(%out : tensor<4x4xf32>) {
  ^bb0(%lhs: tensor<4x4xf32>, %rhs: tensor<4x4xf32>, %dest: tensor<4x4xf32>):
    %mm = linalg.matmul ins(%lhs, %rhs : tensor<4x4xf32>, tensor<4x4xf32>) outs(%dest : tensor<4x4xf32>) -> tensor<4x4xf32>
    iree_payload.yield %mm : tensor<4x4xf32>
  } -> tensor<4x4xf32>
  return
}

// CHECK-LABEL: func.func @strided_views(
// CHECK: iree_payload.region ins(%{{.*}} : memref<4x4xf32, strided<[16, 2], offset: 7>>) outs(%{{.*}} : memref<4x4xf32, strided<[20, 2], offset: 5>>)
// CHECK-NEXT: ^bb0(%[[SOURCE:.*]]: memref<4x4xf32, strided<[16, 2], offset: 7>>, %[[DEST:.*]]: memref<4x4xf32, strided<[20, 2], offset: 5>>):
// CHECK: linalg.generic
// CHECK-SAME: ins(%[[SOURCE]] : memref<4x4xf32, strided<[16, 2], offset: 7>>) outs(%[[DEST]] : memref<4x4xf32, strided<[20, 2], offset: 5>>)
// CHECK: iree_payload.yield
func.func @strided_views(%a: memref<4x4xf32, strided<[16, 2], offset: 7>>, %b: memref<4x4xf32, strided<[20, 2], offset: 5>>) {
  %input = bufferization.to_tensor %a restrict : memref<4x4xf32, strided<[16, 2], offset: 7>> to tensor<4x4xf32>
  %out = bufferization.to_tensor %b restrict writable : memref<4x4xf32, strided<[20, 2], offset: 5>> to tensor<4x4xf32>
  %r = iree_payload.region ins(%input : tensor<4x4xf32>) outs(%out : tensor<4x4xf32>) {
  ^bb0(%source: tensor<4x4xf32>, %dest: tensor<4x4xf32>):
    %one = arith.constant 1.0 : f32
    %result = linalg.generic {indexing_maps = [affine_map<(i,j)->(i,j)>,affine_map<(i,j)->(i,j)>], iterator_types = ["parallel", "parallel"]} ins(%source : tensor<4x4xf32>) outs(%dest : tensor<4x4xf32>) {
    ^bb0(%v: f32, %unused: f32):
      %sum = arith.addf %v, %one : f32
      linalg.yield %sum : f32
    } -> tensor<4x4xf32>
    iree_payload.yield %result : tensor<4x4xf32>
  } -> tensor<4x4xf32>
  return
}

// CHECK-LABEL: func.func @dynamic_views(
// CHECK: iree_payload.region ins(%{{.*}}, %{{.*}} : memref<?x4xf32, strided<[?, 2], offset: ?>>, f32) outs(%{{.*}} : memref<?x4xf32, strided<[?, 2], offset: ?>>)
// CHECK-NEXT: ^bb0(%[[SOURCE:.*]]: memref<?x4xf32, strided<[?, 2], offset: ?>>, %[[SCALE:.*]]: f32, %[[DEST:.*]]: memref<?x4xf32, strided<[?, 2], offset: ?>>):
// CHECK: linalg.generic
// CHECK-SAME: ins(%[[SOURCE]] : memref<?x4xf32, strided<[?, 2], offset: ?>>) outs(%[[DEST]] : memref<?x4xf32, strided<[?, 2], offset: ?>>)
// CHECK: arith.mulf %{{.*}}, %[[SCALE]]
// CHECK: iree_payload.yield
// CHECK-NEXT: } {test.schedule = "preserved"}
func.func @dynamic_views(%a: memref<?x4xf32, strided<[?, 2], offset: ?>>, %b: memref<?x4xf32, strided<[?, 2], offset: ?>>, %scale: f32) {
  %input = bufferization.to_tensor %a restrict : memref<?x4xf32, strided<[?, 2], offset: ?>> to tensor<?x4xf32>
  %out = bufferization.to_tensor %b restrict writable : memref<?x4xf32, strided<[?, 2], offset: ?>> to tensor<?x4xf32>
  %r = iree_payload.region ins(%input, %scale : tensor<?x4xf32>, f32) outs(%out : tensor<?x4xf32>) {
  ^bb0(%source: tensor<?x4xf32>, %s: f32, %dest: tensor<?x4xf32>):
    %result = linalg.generic {indexing_maps = [affine_map<(i,j)->(i,j)>,affine_map<(i,j)->(i,j)>], iterator_types = ["parallel", "parallel"]} ins(%source : tensor<?x4xf32>) outs(%dest : tensor<?x4xf32>) {
    ^bb0(%v: f32, %unused: f32):
      %product = arith.mulf %v, %s : f32
      linalg.yield %product : f32
    } -> tensor<?x4xf32>
    iree_payload.yield %result : tensor<?x4xf32>
  } -> tensor<?x4xf32> {test.schedule = "preserved"}
  return
}

// CHECK-LABEL: func.func @repeated_inputs(
// CHECK-SAME: %[[INPUT:.*]]: memref<4xf32>, %[[OUTPUT:.*]]: memref<4xf32>)
// CHECK-NEXT: iree_payload.region ins(%[[INPUT]], %[[INPUT]] : memref<4xf32>, memref<4xf32>) outs(%[[OUTPUT]] : memref<4xf32>)
// CHECK: linalg.generic
// CHECK: arith.addf
// CHECK: iree_payload.yield
func.func @repeated_inputs(%a: memref<4xf32>, %b: memref<4xf32>) {
  %input = bufferization.to_tensor %a restrict : memref<4xf32> to tensor<4xf32>
  %out = bufferization.to_tensor %b restrict writable : memref<4xf32> to tensor<4xf32>
  %r = iree_payload.region ins(%input, %input : tensor<4xf32>, tensor<4xf32>) outs(%out : tensor<4xf32>) {
  ^bb0(%xarg: tensor<4xf32>, %yarg: tensor<4xf32>, %dest: tensor<4xf32>):
    %result = linalg.generic {indexing_maps = [affine_map<(i)->(i)>,affine_map<(i)->(i)>,affine_map<(i)->(i)>], iterator_types = ["parallel"]} ins(%xarg, %yarg : tensor<4xf32>, tensor<4xf32>) outs(%dest : tensor<4xf32>) {
    ^bb0(%x: f32, %y: f32, %unused: f32):
      %sum = arith.addf %x, %y : f32
      linalg.yield %sum : f32
    } -> tensor<4xf32>
    iree_payload.yield %result : tensor<4xf32>
  } -> tensor<4xf32>
  return
}
