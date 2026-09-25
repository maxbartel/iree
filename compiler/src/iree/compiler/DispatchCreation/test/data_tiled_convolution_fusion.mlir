// RUN: iree-opt --pass-pipeline='builtin.module(func.func(iree-dispatch-creation-form-dispatch-regions))' %s | FileCheck %s --check-prefix=DEFAULT
// RUN: iree-opt --pass-pipeline='builtin.module(func.func(iree-dispatch-creation-form-dispatch-regions{fuse-data-tiled-convolution=true}))' %s | FileCheck %s --check-prefix=FUSED

#input = affine_map<(n, oc, h, w, ic, kh, kw, o, i) -> (n, ic, h + kh, w + kw, i)>
#filter = affine_map<(n, oc, h, w, ic, kh, kw, o, i) -> (oc, ic, kh, kw, i, o)>
#output = affine_map<(n, oc, h, w, ic, kh, kw, o, i) -> (n, oc, h, w, o)>

// DEFAULT-LABEL: func.func @conv_unpack
// DEFAULT: %[[CONV:.*]] = linalg.generic
// DEFAULT: arith.mulf
// DEFAULT: flow.return %[[CONV]]
// DEFAULT: linalg.unpack
// FUSED-LABEL: func.func @conv_unpack
// FUSED: %[[CONV:.*]] = linalg.generic
// FUSED: arith.mulf
// FUSED-NOT: flow.return
// FUSED: %[[UNPACK:.*]] = linalg.unpack %[[CONV]]
// FUSED: flow.return %[[UNPACK]]
func.func @conv_unpack(%a: tensor<2x1x16x16x8xf32>, %b: tensor<2x1x3x3x8x8xf32>) -> tensor<2x14x14x16xf32> {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<2x2x14x14x8xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<2x2x14x14x8xf32>) -> tensor<2x2x14x14x8xf32>
  %conv = linalg.generic {indexing_maps = [#input, #filter, #output],
    iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "parallel", "reduction"]}
    ins(%a, %b : tensor<2x1x16x16x8xf32>, tensor<2x1x3x3x8x8xf32>) outs(%init : tensor<2x2x14x14x8xf32>) {
  ^bb0(%lhs: f32, %rhs: f32, %acc: f32):
    %mul = arith.mulf %lhs, %rhs : f32
    %sum = arith.addf %mul, %acc : f32
    linalg.yield %sum : f32
  } -> tensor<2x2x14x14x8xf32>
  %dest = tensor.empty() : tensor<2x14x14x16xf32>
  %unpack = linalg.unpack %conv outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [8] into %dest : tensor<2x2x14x14x8xf32> -> tensor<2x14x14x16xf32>
  return %unpack : tensor<2x14x14x16xf32>
}

// DEFAULT-LABEL: func.func @permuted_unpack
// DEFAULT: %[[CONV:.*]] = linalg.generic
// DEFAULT: arith.mulf
// DEFAULT: flow.return %[[CONV]]
// DEFAULT: linalg.unpack
// FUSED-LABEL: func.func @permuted_unpack
// FUSED: %[[CONV:.*]] = linalg.generic
// FUSED: arith.mulf
// FUSED: flow.return %[[CONV]]
// FUSED: linalg.unpack
func.func @permuted_unpack(%a: tensor<2x1x16x16x8xf32>, %b: tensor<2x1x3x3x8x8xf32>) -> tensor<2x14x14x16xf32> {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<2x2x14x14x8xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<2x2x14x14x8xf32>) -> tensor<2x2x14x14x8xf32>
  %conv = linalg.generic {indexing_maps = [#input, #filter, #output],
    iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "parallel", "reduction"]}
    ins(%a, %b : tensor<2x1x16x16x8xf32>, tensor<2x1x3x3x8x8xf32>) outs(%init : tensor<2x2x14x14x8xf32>) {
  ^bb0(%lhs: f32, %rhs: f32, %acc: f32):
    %mul = arith.mulf %lhs, %rhs : f32
    %sum = arith.addf %mul, %acc : f32
    linalg.yield %sum : f32
  } -> tensor<2x2x14x14x8xf32>
  %dest = tensor.empty() : tensor<2x14x14x16xf32>
  %unpack = linalg.unpack %conv outer_dims_perm = [3, 0, 1, 2] inner_dims_pos = [3] inner_tiles = [8] into %dest : tensor<2x2x14x14x8xf32> -> tensor<2x14x14x16xf32>
  return %unpack : tensor<2x14x14x16xf32>
}

// DEFAULT-LABEL: func.func @nonunit_batch_collapse
// DEFAULT: %[[CONV:.*]] = linalg.generic
// DEFAULT: arith.mulf
// DEFAULT: flow.return %[[CONV]]
// DEFAULT: linalg.unpack
// FUSED-LABEL: func.func @nonunit_batch_collapse
// FUSED: %[[CONV:.*]] = linalg.generic
// FUSED: arith.mulf
// FUSED: flow.return %[[CONV]]
// FUSED: linalg.unpack
func.func @nonunit_batch_collapse(%a: tensor<2x1x16x16x8xf32>, %b: tensor<2x1x3x3x8x8xf32>) -> tensor<14x14x32xf32> {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<2x2x14x14x8xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<2x2x14x14x8xf32>) -> tensor<2x2x14x14x8xf32>
  %conv = linalg.generic {indexing_maps = [#input, #filter, #output],
    iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "parallel", "reduction"]}
    ins(%a, %b : tensor<2x1x16x16x8xf32>, tensor<2x1x3x3x8x8xf32>) outs(%init : tensor<2x2x14x14x8xf32>) {
  ^bb0(%lhs: f32, %rhs: f32, %acc: f32):
    %mul = arith.mulf %lhs, %rhs : f32
    %sum = arith.addf %mul, %acc : f32
    linalg.yield %sum : f32
  } -> tensor<2x2x14x14x8xf32>
  %collapsed = tensor.collapse_shape %conv [[0, 1], [2], [3], [4]] : tensor<2x2x14x14x8xf32> into tensor<4x14x14x8xf32>
  %dest = tensor.empty() : tensor<14x14x32xf32>
  %unpack = linalg.unpack %collapsed outer_dims_perm = [2, 0, 1] inner_dims_pos = [2] inner_tiles = [8] into %dest : tensor<4x14x14x8xf32> -> tensor<14x14x32xf32>
  return %unpack : tensor<14x14x32xf32>
}
