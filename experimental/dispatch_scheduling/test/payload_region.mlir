// RUN: iree-opt --iree-plugin=dispatch_scheduling %s | FileCheck %s

// CHECK-LABEL: func.func @minimal_payload_region
func.func @minimal_payload_region() {
  // CHECK:      iree_payload.region {
  // CHECK-NEXT:   iree_payload.yield
  // CHECK-NEXT: }
  iree_payload.region {
    iree_payload.yield
  }
  return
}

// CHECK-LABEL: func.func @tensor_dps_payload_region
func.func @tensor_dps_payload_region(
    %arg0: tensor<4xf32>, %init0: tensor<4xf32>, %init1: tensor<4xf32>)
    -> (tensor<4xf32>, tensor<4xf32>) {
  // CHECK:      %[[R:.*]]:2 = iree_payload.region
  // CHECK-SAME:     ins(%{{.*}} : tensor<4xf32>)
  // CHECK-SAME:     outs(%{{.*}}, %{{.*}} : tensor<4xf32>, tensor<4xf32>)
  // CHECK:      ^bb0(%{{.*}}: tensor<4xf32>, %{{.*}}: tensor<4xf32>, %{{.*}}: tensor<4xf32>):
  // CHECK-NEXT:   iree_payload.yield %{{.*}}, %{{.*}} : tensor<4xf32>, tensor<4xf32>
  // CHECK-NEXT: } -> tensor<4xf32>, tensor<4xf32>
  %0, %1 = iree_payload.region
      ins(%arg0 : tensor<4xf32>)
      outs(%init0, %init1 : tensor<4xf32>, tensor<4xf32>) {
  ^bb0(%arg0_b: tensor<4xf32>, %init0_b: tensor<4xf32>, %init1_b: tensor<4xf32>):
    iree_payload.yield %init0_b, %init1_b : tensor<4xf32>, tensor<4xf32>
  } -> tensor<4xf32>, tensor<4xf32>
  return %0, %1 : tensor<4xf32>, tensor<4xf32>
}

// CHECK-LABEL: func.func @memref_dps_payload_region
func.func @memref_dps_payload_region(
    %arg0: memref<4xf32>, %out0: memref<4xf32>, %out1: memref<4xf32>) {
  // CHECK:      iree_payload.region
  // CHECK-SAME:     ins(%{{.*}} : memref<4xf32>)
  // CHECK-SAME:     outs(%{{.*}}, %{{.*}} : memref<4xf32>, memref<4xf32>)
  // CHECK:      ^bb0(%{{.*}}: memref<4xf32>, %{{.*}}: memref<4xf32>, %{{.*}}: memref<4xf32>):
  // CHECK-NEXT:   iree_payload.yield
  // CHECK-NEXT: }
  iree_payload.region
      ins(%arg0 : memref<4xf32>)
      outs(%out0, %out1 : memref<4xf32>, memref<4xf32>) {
  ^bb0(%arg0_b: memref<4xf32>, %out0_b: memref<4xf32>, %out1_b: memref<4xf32>):
    iree_payload.yield
  }
  return
}
