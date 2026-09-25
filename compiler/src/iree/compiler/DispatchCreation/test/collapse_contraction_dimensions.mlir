// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(util.func(iree-dispatch-creation-collapse-contraction-dimensions,iree-dispatch-creation-annotate-data-tiling-hints,iree-dispatch-creation-set-encoding))' %s | FileCheck %s

// Before normalization this contraction is rejected by encoding annotation:
// it has two M dimensions. Preserve the function's original result shape.
// CHECK-LABEL: @multi_m(
// CHECK: tensor.collapse_shape {{.*}} : tensor<2x3x8xf32> into tensor<6x8xf32>
// CHECK: iree_encoding.set_encoding {{.*}} : tensor<6x8xf32> -> tensor<6x8xf32, #
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction"]
// CHECK-SAME: tensor<6x4xf32, #
// CHECK: tensor.expand_shape {{.*}} output_shape [2, 3, 4] : tensor<6x4xf32> into tensor<2x3x4xf32>
// CHECK: util.return {{.*}} : tensor<2x3x4xf32>
util.func public @multi_m(%a: tensor<2x3x8xf32>, %b: tensor<8x4xf32>, %init: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
  %result = linalg.generic {
    indexing_maps = [affine_map<(m0, m1, n, k) -> (m0, m1, k)>,
                     affine_map<(m0, m1, n, k) -> (k, n)>,
                     affine_map<(m0, m1, n, k) -> (m0, m1, n)>],
    iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
    ins(%a, %b : tensor<2x3x8xf32>, tensor<8x4xf32>)
    outs(%init : tensor<2x3x4xf32>) {
  ^bb0(%lhs: f32, %rhs: f32, %acc: f32):
    %mul = arith.mulf %lhs, %rhs : f32
    %sum = arith.addf %mul, %acc : f32
    linalg.yield %sum : f32
  } -> tensor<2x3x4xf32>
  util.return %result : tensor<2x3x4xf32>
}

// -----

// CHECK-LABEL: @multi_k_dim_generic(
// CHECK: tensor.collapse_shape {{.*}} : tensor<256x64x2xf32> into tensor<256x128xf32>
// CHECK: tensor.collapse_shape {{.*}} : tensor<64x2x512xf32> into tensor<128x512xf32>
// CHECK: iree_encoding.set_encoding {{.*}} : tensor<256x128xf32> -> tensor<256x128xf32, #
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction"]
// CHECK: util.return {{.*}} : tensor<256x512xf32>
util.func public @multi_k_dim_generic(%arg0: tensor<256x64x2xf32>, %arg1: tensor<64x2x512xf32>,
                                      %arg2: tensor<256x512xf32>) -> tensor<256x512xf32> {
    %4 = linalg.generic {
        indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>,
                         affine_map<(d0, d1, d2, d3) -> (d2, d3, d1)>,
                         affine_map<(d0, d1, d2, d3) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel", "reduction", "reduction"]}
        ins(%arg0, %arg1: tensor<256x64x2xf32>, tensor<64x2x512xf32>) outs(%arg2: tensor<256x512xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %5 = arith.mulf %in, %in_0 : f32
      %6 = arith.addf %5, %out : f32
      linalg.yield %6 : f32
    } -> tensor<256x512xf32>
  util.return %4 : tensor<256x512xf32>
}

// -----

// CHECK-LABEL: @multi_batch_dim_generic(
// CHECK: iree_encoding.set_encoding {{.*}} : tensor<32x256x128xf32> -> tensor<32x256x128xf32, #
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "reduction"]
// CHECK: tensor.expand_shape {{.*}} output_shape [4, 8, 256, 512]
// CHECK: util.return {{.*}} : tensor<4x8x256x512xf32>
util.func public @multi_batch_dim_generic(%arg0: tensor<4x8x256x128xf32>, %arg1: tensor<4x8x128x512xf32>,
                                          %arg2: tensor<4x8x256x512xf32>) -> tensor<4x8x256x512xf32> {
    %4 = linalg.generic {
        indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>,
                         affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>,
                         affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>],
        iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]}
        ins(%arg0, %arg1: tensor<4x8x256x128xf32>, tensor<4x8x128x512xf32>) outs(%arg2: tensor<4x8x256x512xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %5 = arith.mulf %in, %in_0 : f32
      %6 = arith.addf %5, %out : f32
      linalg.yield %6 : f32
    } -> tensor<4x8x256x512xf32>
  util.return %4 : tensor<4x8x256x512xf32>
}

// -----

// CHECK-LABEL: @broadcast_rhs_batch_mmt(
// CHECK: iree_encoding.set_encoding {{.*}} : tensor<16384x1280xi8> -> tensor<16384x1280xi8, #
// CHECK: linalg.generic
// CHECK-SAME: tensor<16384x10240xi32, #
// CHECK: arith.extsi
// CHECK: arith.extsi
// CHECK: arith.muli
// CHECK: arith.addi
// CHECK: tensor.expand_shape {{.*}} output_shape [16, 1024, 10240]
util.func public @broadcast_rhs_batch_mmt(%arg0: tensor<16x1024x1280xi8>, %arg1: tensor<10240x1280xi8>,
                                          %arg2: tensor<16x1024x10240xi32>) -> tensor<16x1024x10240xi32> {
  %20 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,
                                         affine_map<(d0, d1, d2, d3) -> (d2, d3)>,
                                         affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>],
                        iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
    ins(%arg0, %arg1: tensor<16x1024x1280xi8>, tensor<10240x1280xi8>) outs(%arg2 : tensor<16x1024x10240xi32>) {
  ^bb0(%in: i8, %in_0: i8, %acc: i32):
    %22 = arith.extsi %in : i8 to i32
    %23 = arith.extsi %in_0 : i8 to i32
    %24 = arith.muli %22, %23 : i32
    %25 = arith.addi %acc, %24 : i32
    linalg.yield %25 : i32
  } -> tensor<16x1024x10240xi32>
  util.return %20: tensor<16x1024x10240xi32>
}

// -----

// CHECK-LABEL: @dynamic_multi_m(
// CHECK: %[[DIM:.*]] = tensor.dim
// CHECK: tensor.collapse_shape {{.*}} : tensor<?x3x8xf32> into tensor<?x8xf32>
// CHECK: iree_encoding.set_encoding {{.*}} encoding_dims{{.*}} : tensor<?x8xf32> -> tensor<?x8xf32, #
// CHECK: linalg.generic
// CHECK-SAME: tensor<?x4xf32, #
// CHECK: tensor.expand_shape {{.*}} output_shape [%[[DIM]], 3, 4]
// CHECK: util.return {{.*}} : tensor<?x3x4xf32>
util.func public @dynamic_multi_m(%a: tensor<?x3x8xf32>, %b: tensor<8x4xf32>, %init: tensor<?x3x4xf32>) -> tensor<?x3x4xf32> {
  %result = linalg.generic {
    indexing_maps = [affine_map<(m0, m1, n, k) -> (m0, m1, k)>,
                     affine_map<(m0, m1, n, k) -> (k, n)>,
                     affine_map<(m0, m1, n, k) -> (m0, m1, n)>],
    iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
    ins(%a, %b : tensor<?x3x8xf32>, tensor<8x4xf32>)
    outs(%init : tensor<?x3x4xf32>) {
  ^bb0(%lhs: f32, %rhs: f32, %acc: f32):
    %mul = arith.mulf %lhs, %rhs : f32
    %sum = arith.addf %mul, %acc : f32
    linalg.yield %sum : f32
  } -> tensor<?x3x4xf32>
  util.return %result : tensor<?x3x4xf32>
}

// -----

// M dimensions are not contiguous in the lhs; preserve baseline fallback.
// CHECK-LABEL: @noncontiguous_m(
// CHECK-NOT: tensor.collapse_shape
// CHECK-NOT: iree_encoding
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "reduction"]
// CHECK-NOT: iree_encoding
// CHECK: util.return {{.*}} : tensor<2x3x4xf32>
util.func public @noncontiguous_m(%a: tensor<2x8x3xf32>, %b: tensor<8x4xf32>, %init: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
  %result = linalg.generic {
    indexing_maps = [affine_map<(m0, m1, n, k) -> (m0, k, m1)>,
                     affine_map<(m0, m1, n, k) -> (k, n)>,
                     affine_map<(m0, m1, n, k) -> (m0, m1, n)>],
    iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
    ins(%a, %b : tensor<2x8x3xf32>, tensor<8x4xf32>)
    outs(%init : tensor<2x3x4xf32>) {
  ^bb0(%lhs: f32, %rhs: f32, %acc: f32):
    %mul = arith.mulf %lhs, %rhs : f32
    %sum = arith.addf %mul, %acc : f32
    linalg.yield %sum : f32
  } -> tensor<2x3x4xf32>
  util.return %result : tensor<2x3x4xf32>
}
