// RUN: iree-opt --split-input-file --iree-dispatch-creation-propagate-data-tiling-encodings %s | FileCheck %s

// The same encoding rules as hoist_encoding_ops, applied before dispatch formation.
// The late pass must keep its original placement policy on these inputs.
// RUN: iree-opt --split-input-file --iree-dispatch-creation-hoist-encoding-ops %s | FileCheck %s --check-prefix=LATE

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#encoding = #iree_encoding.encoding<operand_index = 1 : index, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map2, #map3, #map4]>
// CHECK-LABEL: @bubble_through_dequant(
// CHECK: %[[A:.*]] = iree_encoding.set_encoding %arg0
// CHECK: %[[B:.*]] = iree_encoding.set_encoding %arg1
// CHECK: %[[C:.*]] = iree_encoding.set_encoding %arg2
// CHECK: linalg.generic {{.*}} ins(%[[A]], %[[B]], %[[C]]
// CHECK-SAME: tensor<2x11008x128xi8, #
// CHECK-NOT: iree_encoding.set_encoding
// CHECK: util.return
// LATE-LABEL: @bubble_through_dequant(
// LATE: %[[DEQUANT:.*]] = linalg.generic
// LATE: iree_encoding.set_encoding %[[DEQUANT]]
util.func public @bubble_through_dequant(
    %arg0: tensor<2x11008x128xi8>, %arg1: tensor<2x11008xf32>, %arg2: tensor<2x11008xf32>) -> tensor<2x11008x128xf32, #encoding> {
    %8 = tensor.empty() : tensor<2x11008x128xf32>
    %11 = linalg.generic
        {indexing_maps = [#map, #map1, #map1, #map],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%arg0, %arg1, %arg2 : tensor<2x11008x128xi8>, tensor<2x11008xf32>, tensor<2x11008xf32>)
        outs(%8 : tensor<2x11008x128xf32>) {
    ^bb0(%in: i8, %in_0: f32, %in_1: f32, %out: f32):
      %18 = arith.extui %in : i8 to i32
      %19 = arith.uitofp %18 : i32 to f32
      %20 = arith.subf %19, %in_1 : f32
      %21 = arith.mulf %20, %in_0 : f32
      linalg.yield %21 : f32
    } -> tensor<2x11008x128xf32>
    %13 = iree_encoding.set_encoding %11 : tensor<2x11008x128xf32> -> tensor<2x11008x128xf32, #encoding>
    util.return %13 : tensor<2x11008x128xf32, #encoding>
}

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#encoding = #iree_encoding.encoding<operand_index = 1 : index, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>]>
// CHECK-LABEL: @bubble_through_broadcast(
// CHECK: %[[A:.*]] = iree_encoding.set_encoding %arg0 : tensor<11008x128xf32> -> tensor<11008x128xf32, #
// CHECK: linalg.generic {{.*}} ins(%[[A]]
// CHECK-SAME: outs(%{{.*}} : tensor<2x11008x128xf32, #
// CHECK-NOT: iree_encoding.set_encoding
// CHECK: util.return
util.func public @bubble_through_broadcast(
    %arg0: tensor<11008x128xf32>) -> tensor<2x11008x128xf32, #encoding> {
    %8 = tensor.empty() : tensor<2x11008x128xf32>
    %11 = linalg.generic
        {indexing_maps = [#map1, #map],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%arg0 : tensor<11008x128xf32>)
        outs(%8 : tensor<2x11008x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<2x11008x128xf32>
    %13 = iree_encoding.set_encoding %11 : tensor<2x11008x128xf32> -> tensor<2x11008x128xf32, #encoding>
    util.return %13 : tensor<2x11008x128xf32, #encoding>
}

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
#map4 = affine_map<(d0, d1) -> ()>
#encoding = #iree_encoding.encoding<operand_index = 2 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>
// CHECK-LABEL: @propagate_unset_encoding_through_generic_with_scalar(
// CHECK-NOT: iree_encoding.unset_encoding
// CHECK: %[[RESULT:.*]] = linalg.generic
// CHECK-SAME: ins(%arg0, %arg1 : tensor<4096x?xf32, #{{.*}}>, f32)
// CHECK: iree_encoding.unset_encoding %[[RESULT]]
// LATE-LABEL: @propagate_unset_encoding_through_generic_with_scalar(
// LATE: %[[RAW:.*]] = iree_encoding.unset_encoding
// LATE: linalg.generic {{.*}} ins(%[[RAW]],
util.func public @propagate_unset_encoding_through_generic_with_scalar(%arg0: tensor<4096x?xf32, #encoding>, %arg1: f32, %arg2: index) -> tensor<4096x?xf32> {
    %1 = iree_encoding.unset_encoding %arg0 : tensor<4096x?xf32, #encoding> -> tensor<4096x?xf32>{%arg2}
    %2 = tensor.empty(%arg2) : tensor<4096x?xf32>
    %3 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel"]} ins(%1, %arg1 : tensor<4096x?xf32>, f32) outs(%2 : tensor<4096x?xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %4 = arith.mulf %in, %in_0 : f32
      linalg.yield %4 : f32
    } -> tensor<4096x?xf32>
    util.return %3 : tensor<4096x?xf32>
}

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
#map4 = affine_map<(d0, d1) -> ()>
#encoding = #iree_encoding.encoding<operand_index = 2 : index, op_type =  matmul, element_types = [f8E4M3FNUZ, f8E4M3FNUZ, f32], user_indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>]>
// CHECK-LABEL: @dont_propagate_unset_encoding_with_multiple_uses(
// CHECK: %[[RAW:.*]] = iree_encoding.unset_encoding
// CHECK: linalg.generic {{.*}} ins(%[[RAW]]
// CHECK-NOT: iree_encoding
// CHECK: util.return
util.func public @dont_propagate_unset_encoding_with_multiple_uses(%arg0: tensor<?x4096xf32, #encoding>, %arg1: tensor<f32>, %arg2: index) -> (tensor<?x4096xbf16>, tensor<?x4096xbf16>) {
    %1 = iree_encoding.unset_encoding %arg0 : tensor<?x4096xf32, #encoding> -> tensor<?x4096xf32>{%arg2}
    %2 = tensor.empty(%arg2) : tensor<?x4096xbf16>
    %3 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel"]} ins(%1, %arg1 : tensor<?x4096xf32>, tensor<f32>) outs(%2 : tensor<?x4096xbf16>) {
    ^bb0(%in: f32, %in_0: f32, %out: bf16):
      %5 = arith.mulf %in, %in_0 : f32
      %6 = arith.truncf %5 : f32 to bf16
      linalg.yield %6 : bf16
    } -> tensor<?x4096xbf16>
    %4 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel"]} ins(%1, %arg1 : tensor<?x4096xf32>, tensor<f32>) outs(%2 : tensor<?x4096xbf16>) {
    ^bb0(%in: f32, %in_0: f32, %out: bf16):
      %5 = arith.addf %in, %in_0 : f32
      %6 = arith.truncf %5 : f32 to bf16
      linalg.yield %6 : bf16
    } -> tensor<?x4096xbf16>
    util.return %3, %4 : tensor<?x4096xbf16>, tensor<?x4096xbf16>
}

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#encoding = #iree_encoding.encoding<operand_index = 1 : index, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map2, #map3, #map4], iteration_sizes = [2, ?, ?, ?]>
// CHECK-LABEL: @bubble_with_rematerialized_encoding_dims(
// CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG: %[[C11008:.*]] = arith.constant 11008 : index
// CHECK-DAG: %[[C128:.*]] = arith.constant 128 : index
// CHECK: iree_encoding.set_encoding %arg0 encoding_dims{%[[C2]], %[[C11008]], %[[C128]]}
// CHECK: iree_encoding.set_encoding %arg1 encoding_dims{%[[C2]], %[[C11008]], %[[C128]]}
// CHECK: iree_encoding.set_encoding %arg2 encoding_dims{%[[C2]], %[[C11008]], %[[C128]]}
// CHECK: linalg.generic
// CHECK-NOT: iree_encoding.set_encoding
// CHECK: util.return
util.func public @bubble_with_rematerialized_encoding_dims(
    %arg0: tensor<2x11008x128xi8>, %arg1: tensor<2x11008xf32>, %arg2: tensor<2x11008xf32>) -> tensor<2x11008x128xf32, #encoding> {
    %8 = tensor.empty() : tensor<2x11008x128xf32>
    %11 = linalg.generic
        {indexing_maps = [#map, #map1, #map1, #map],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%arg0, %arg1, %arg2 : tensor<2x11008x128xi8>, tensor<2x11008xf32>, tensor<2x11008xf32>)
        outs(%8 : tensor<2x11008x128xf32>) {
    ^bb0(%in: i8, %in_0: f32, %in_1: f32, %out: f32):
      %18 = arith.extui %in : i8 to i32
      %19 = arith.uitofp %18 : i32 to f32
      %20 = arith.subf %19, %in_1 : f32
      %21 = arith.mulf %20, %in_0 : f32
      linalg.yield %21 : f32
    } -> tensor<2x11008x128xf32>
    // encoding_dims are computed from the output, but can be rematerialized
    // from the first input (which then fold to constants).
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %m = tensor.dim %11, %c0 : tensor<2x11008x128xf32>
    %n = tensor.dim %11, %c1 : tensor<2x11008x128xf32>
    %k = tensor.dim %11, %c2 : tensor<2x11008x128xf32>
    %13 = iree_encoding.set_encoding %11 encoding_dims{%m, %n, %k} : tensor<2x11008x128xf32> -> tensor<2x11008x128xf32, #encoding>
    util.return %13 : tensor<2x11008x128xf32, #encoding>
}

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
#map4 = affine_map<(d0, d1) -> (d0)>
#encoding = #iree_encoding.encoding<operand_index = 2 : index, op_type =  matmul, element_types = [f8E4M3FNUZ, f8E4M3FNUZ, f32], user_indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>]>
// CHECK-LABEL: @dont_propagate_unset_encoding_through_generic_with_reduction(
// CHECK: %[[RAW:.*]] = iree_encoding.unset_encoding
// CHECK: linalg.generic {{.*}} ins(%[[RAW]]
// CHECK-NOT: iree_encoding
// CHECK: util.return
util.func public @dont_propagate_unset_encoding_through_generic_with_reduction(%arg0: tensor<?x4096xf32, #encoding>, %arg1: index) -> tensor<?xf32> {
    %1 = iree_encoding.unset_encoding %arg0 : tensor<?x4096xf32, #encoding> -> tensor<?x4096xf32>{%arg1}
    %2 = tensor.empty(%arg1) : tensor<?xf32>
    %3 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%1 : tensor<?x4096xf32>) outs(%2 : tensor<?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4 = arith.addf %in, %out : f32
      linalg.yield %4 : f32
    } -> tensor<?xf32>
    util.return %3 : tensor<?xf32>
}

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
#map4 = affine_map<(d0, d1) -> (d0, d1, d0)>
#encoding = #iree_encoding.encoding<operand_index = 2 : index, op_type =  matmul, element_types = [f8E4M3FNUZ, f8E4M3FNUZ, f32], user_indexing_maps = [#map, #map1, #map2]>
// CHECK-LABEL: @dont_propagate_non_projected_permutation(
// CHECK: %[[RAW:.*]] = iree_encoding.unset_encoding
// CHECK: linalg.generic {{.*}} ins(%[[RAW]]
// CHECK-NOT: iree_encoding
// CHECK: util.return
util.func public @dont_propagate_non_projected_permutation(%arg0: tensor<?x4096xf32, #encoding>, %arg1: tensor<?x4x4096xf32>, %arg2: index) -> tensor<?x4096xf32> {
    %1 = iree_encoding.unset_encoding %arg0 : tensor<?x4096xf32, #encoding> -> tensor<?x4096xf32>{%arg2}
    %2 = tensor.empty(%arg2) : tensor<?x4096xf32>
    %3 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel"]} ins(%1, %arg1 : tensor<?x4096xf32>, tensor<?x4x4096xf32>) outs(%2 : tensor<?x4096xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %4 = arith.addf %in, %in_0 : f32
      %5 = arith.addf %4, %out : f32
      linalg.yield %5 : f32
    } -> tensor<?x4096xf32>
    util.return %3 : tensor<?x4096xf32>
}

// -----

#map_dom = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map_dom1 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map_dom2 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map_dom3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map_dom4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#encoding_dom = #iree_encoding.encoding<operand_index = 1 : index, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map_dom2, #map_dom3, #map_dom4], iteration_sizes = [2, ?, ?, ?]>
// CHECK-LABEL: @bubble_with_recursive_rematerialization(
// CHECK: %[[DIM:.*]] = tensor.dim %arg3
// CHECK: iree_encoding.set_encoding %arg0 encoding_dims{%[[DIM]],
// CHECK: iree_encoding.set_encoding %arg1 encoding_dims{%[[DIM]],
// CHECK: iree_encoding.set_encoding %arg2 encoding_dims{%[[DIM]],
// CHECK: linalg.generic
// CHECK-NOT: iree_encoding.set_encoding
// CHECK: util.return
util.func public @bubble_with_recursive_rematerialization(
    %arg0: tensor<2x11008x128xi8>, %arg1: tensor<2x11008xf32>, %arg2: tensor<2x11008xf32>,
    %dominating: tensor<?x?xf32>) -> tensor<2x11008x128xf32, #encoding_dom> {
    %8 = tensor.empty() : tensor<2x11008x128xf32>
    %11 = linalg.generic
        {indexing_maps = [#map_dom, #map_dom1, #map_dom1, #map_dom],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%arg0, %arg1, %arg2 : tensor<2x11008x128xi8>, tensor<2x11008xf32>, tensor<2x11008xf32>)
        outs(%8 : tensor<2x11008x128xf32>) {
    ^bb0(%in: i8, %in_0: f32, %in_1: f32, %out: f32):
      %18 = arith.extui %in : i8 to i32
      %19 = arith.uitofp %18 : i32 to f32
      %20 = arith.subf %19, %in_1 : f32
      %21 = arith.mulf %20, %in_0 : f32
      linalg.yield %21 : f32
    } -> tensor<2x11008x128xf32>
    // encoding_dim uses tensor.dim on a dominating tensor - can be rematerialized
    %c0 = arith.constant 0 : index
    %m = tensor.dim %dominating, %c0 : tensor<?x?xf32>
    %c11008 = arith.constant 11008 : index
    %c128 = arith.constant 128 : index
    %13 = iree_encoding.set_encoding %11 encoding_dims{%m, %c11008, %c128} : tensor<2x11008x128xf32> -> tensor<2x11008x128xf32, #encoding_dom>
    util.return %13 : tensor<2x11008x128xf32, #encoding_dom>
}

// -----

#map_sink = affine_map<(d0, d1, d2) -> (d0, d2)>
#map_sink1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map_sink2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map_sink3 = affine_map<(d0, d1) -> (d0, d1)>
#map_sink4 = affine_map<(d0, d1) -> ()>
#encoding_sink = #iree_encoding.encoding<operand_index = 2 : index, op_type = matmul, element_types = [f16, f16, f32], user_indexing_maps = [#map_sink, #map_sink1, #map_sink2], iteration_sizes = [?, 4096, 4096]>
// CHECK-LABEL: @sink_unset_encoding_with_encoding_dims(
// CHECK: %[[SCALE:.*]] = iree_encoding.set_encoding %arg1 encoding_dims{%arg2}
// CHECK: %[[RESULT:.*]] = linalg.generic {{.*}} ins(%arg0, %[[SCALE]]
// CHECK-SAME: tensor<?x4096xbf16, #
// CHECK: iree_encoding.unset_encoding %[[RESULT]] encoding_dims{%arg2}
util.func public @sink_unset_encoding_with_encoding_dims(%arg0: tensor<?x4096xf32, #encoding_sink>, %arg1: tensor<f32>, %m: index) -> tensor<?x4096xbf16> {
    %1 = iree_encoding.unset_encoding %arg0 encoding_dims{%m} : tensor<?x4096xf32, #encoding_sink> -> tensor<?x4096xf32>{%m}
    %2 = tensor.empty(%m) : tensor<?x4096xbf16>
    %3 = linalg.generic {indexing_maps = [#map_sink3, #map_sink4, #map_sink3], iterator_types = ["parallel", "parallel"]} ins(%1, %arg1 : tensor<?x4096xf32>, tensor<f32>) outs(%2 : tensor<?x4096xbf16>) {
    ^bb0(%in: f32, %in_0: f32, %out: bf16):
      %4 = arith.mulf %in, %in_0 : f32
      %5 = arith.truncf %4 : f32 to bf16
      linalg.yield %5 : bf16
    } -> tensor<?x4096xbf16>
    util.return %3 : tensor<?x4096xbf16>
}

// -----

#map_lhs = affine_map<(d0, d1, d2) -> (d0, d2)>
#map_rhs = affine_map<(d0, d1, d2) -> (d1, d2)>
#map_res = affine_map<(d0, d1, d2) -> (d0, d1)>
#map_bcast_in  = affine_map<(d0, d1, d2) -> (d1, d2)>
#map_bcast_out = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#encoding_bcast = #iree_encoding.encoding<operand_index = 2 : index, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map_lhs, #map_rhs, #map_res], iteration_sizes = [2, 2, 2]>
// CHECK-LABEL: @no_sink_unset_encoding_through_broadcast(
// CHECK: %[[RAW:.*]] = iree_encoding.unset_encoding
// CHECK: linalg.generic {{.*}} ins(%[[RAW]] :
// CHECK-NOT: iree_encoding
// CHECK: util.return
util.func public @no_sink_unset_encoding_through_broadcast(%arg0: tensor<2x2xf32, #encoding_bcast>) -> tensor<2x2x2xf32> {
    %1 = iree_encoding.unset_encoding %arg0 : tensor<2x2xf32, #encoding_bcast> -> tensor<2x2xf32>
    %empty = tensor.empty() : tensor<2x2x2xf32>
    %2 = linalg.generic {
        indexing_maps = [#map_bcast_in, #map_bcast_out],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%1 : tensor<2x2xf32>) outs(%empty : tensor<2x2x2xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
    } -> tensor<2x2x2xf32>
    util.return %2 : tensor<2x2x2xf32>
}

// -----

// The early placement policy must leave dispatch-contained encodings alone.
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#encoding = #iree_encoding.encoding<operand_index = 1 : index, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>]>
// CHECK-LABEL: @leave_existing_dispatch(
// CHECK-NOT: iree_encoding.set_encoding
// CHECK: flow.dispatch.region
// CHECK: %[[BCAST:.*]] = linalg.generic
// CHECK: iree_encoding.set_encoding %[[BCAST]]
// CHECK: flow.return
util.func public @leave_existing_dispatch(
    %arg0: tensor<11008x128xf32>) -> tensor<2x11008x128xf32, #encoding> {
  %6 = flow.dispatch.region -> (tensor<2x11008x128xf32, #encoding>) {
    %8 = tensor.empty() : tensor<2x11008x128xf32>
    %11 = linalg.generic
        {indexing_maps = [#map1, #map],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%arg0 : tensor<11008x128xf32>)
        outs(%8 : tensor<2x11008x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<2x11008x128xf32>
    %13 = iree_encoding.set_encoding %11 : tensor<2x11008x128xf32> -> tensor<2x11008x128xf32, #encoding>
    flow.return %13 : tensor<2x11008x128xf32, #encoding>
  }
  util.return %6 : tensor<2x11008x128xf32, #encoding>
}
