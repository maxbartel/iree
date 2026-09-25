// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(util.func(iree-dispatch-creation-collapse-contraction-dimensions))" %s | FileCheck %s --implicit-check-not=tensor.collapse_shape --implicit-check-not=tensor.expand_shape

#compilation = #iree_codegen.compilation_info<
  lowering_config = #iree_codegen.lowering_config<tile_sizes = [[0, 0, 0, 0]]>,
  translation_info = #iree_codegen.translation_info<pipeline = #iree_cpu.pipeline<Default>>>

// A preset lowering configuration refers to the original loop dimensions.
// CHECK-LABEL: @configured(
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "reduction"]
// CHECK-SAME: compilation_info =
// CHECK: util.return {{.*}} : tensor<2x3x4xf32>
util.func public @configured(%a: tensor<2x3x8xf32>, %b: tensor<8x4xf32>, %init: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
  %result = linalg.generic {compilation_info = #compilation,
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

// Dispatch formation already owns normalization inside this region.
// CHECK-LABEL: @in_dispatch(
// CHECK: flow.dispatch.region
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "reduction"]
// CHECK: flow.return {{.*}} : tensor<2x3x4xf32>
util.func public @in_dispatch(%a: tensor<2x3x8xf32>, %b: tensor<8x4xf32>, %init: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
  %dispatch = flow.dispatch.region -> (tensor<2x3x4xf32>) {
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
    flow.return %result : tensor<2x3x4xf32>
  }
  util.return %dispatch : tensor<2x3x4xf32>
}

// -----

// This pass normalizes contractions, not unrelated elementwise operations.
// CHECK-LABEL: @pointwise(
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel"]
// CHECK: math.absf
// CHECK: util.return {{.*}} : tensor<2x3x4xf32>
util.func public @pointwise(%input: tensor<2x3x4xf32>, %init: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
  %result = linalg.generic {
    indexing_maps = [affine_map<(i, j, k) -> (i, j, k)>, affine_map<(i, j, k) -> (i, j, k)>],
    iterator_types = ["parallel", "parallel", "parallel"]}
    ins(%input : tensor<2x3x4xf32>) outs(%init : tensor<2x3x4xf32>) {
  ^bb0(%element: f32, %output: f32):
    %abs = math.absf %element : f32
    linalg.yield %abs : f32
  } -> tensor<2x3x4xf32>
  util.return %result : tensor<2x3x4xf32>
}
