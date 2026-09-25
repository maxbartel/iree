// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(util.func(iree-dispatch-creation-form-dispatch-regions{fuse-mmt4d=true}))" %s | FileCheck %s

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
  // CHECK-LABEL: util.func public @mmt4d_relu_unpack
  // CHECK: %[[LHS:.*]] = flow.dispatch.region
  // CHECK: linalg.pack
  // CHECK: flow.return
  // CHECK: %[[RHS:.*]] = flow.dispatch.region
  // CHECK: linalg.pack
  // CHECK: flow.return
  // CHECK: %[[RESULT:.*]] = flow.dispatch.region -> (tensor<64x64xf32>)
  // CHECK: %[[MM:.*]] = linalg.mmt4d ins(%[[LHS]], %[[RHS]]
  // CHECK-NOT: flow.return
  // CHECK: %[[EPILOGUE:.*]] = linalg.generic
  // CHECK-SAME: ins(%[[MM]] : tensor<8x8x8x8xf32>)
  // CHECK: arith.maximumf
  // CHECK-NOT: flow.return
  // CHECK: %[[UNPACK:.*]] = linalg.unpack
  // CHECK: flow.return %[[UNPACK]] : tensor<64x64xf32>
  // CHECK-NOT: flow.dispatch.region
  // CHECK: util.return %[[RESULT]]
  util.func public @mmt4d_relu_unpack(%arg0: tensor<64x128xf32>, %arg1: tensor<128x64xf32>) -> tensor<64x64xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<8x128x8x1xf32>
    %pack = linalg.pack %arg0 outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [8, 1] into %0 : tensor<64x128xf32> -> tensor<8x128x8x1xf32>
    %pack_0 = linalg.pack %arg1 outer_dims_perm = [1, 0] inner_dims_pos = [1, 0] inner_tiles = [8, 1] into %0 : tensor<128x64xf32> -> tensor<8x128x8x1xf32>
    %1 = tensor.empty() : tensor<8x8x8x8xf32>
    %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xf32>
    %3 = linalg.mmt4d ins(%pack, %pack_0 : tensor<8x128x8x1xf32>, tensor<8x128x8x1xf32>) outs(%2 : tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xf32>
    %4 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3 : tensor<8x8x8x8xf32>) outs(%1 : tensor<8x8x8x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %6 = arith.maximumf %in, %cst : f32
      linalg.yield %6 : f32
    } -> tensor<8x8x8x8xf32>
    %5 = tensor.empty() : tensor<64x64xf32>
    %unpack = linalg.unpack %4 outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [8, 8] into %5 : tensor<8x8x8x8xf32> -> tensor<64x64xf32>
    util.return %unpack : tensor<64x64xf32>
  }

// -----

#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
  // CHECK-LABEL: util.func public @batch_mmt4d_relu_unpack
  // CHECK: %[[LHS:.*]] = flow.dispatch.region
  // CHECK: linalg.pack
  // CHECK: flow.return
  // CHECK: %[[RHS:.*]] = flow.dispatch.region
  // CHECK: linalg.pack
  // CHECK: flow.return
  // CHECK: %[[RESULT:.*]] = flow.dispatch.region -> (tensor<2x64x64xf32>)
  // CHECK: %[[MM:.*]] = linalg.batch_mmt4d ins(%[[LHS]], %[[RHS]]
  // CHECK-NOT: flow.return
  // CHECK: %[[EPILOGUE:.*]] = linalg.generic
  // CHECK-SAME: ins(%[[MM]] : tensor<2x8x8x8x8xf32>)
  // CHECK: arith.maximumf
  // CHECK-NOT: flow.return
  // CHECK: %[[UNPACK:.*]] = linalg.unpack
  // CHECK: flow.return %[[UNPACK]] : tensor<2x64x64xf32>
  // CHECK-NOT: flow.dispatch.region
  // CHECK: util.return %[[RESULT]]
  util.func public @batch_mmt4d_relu_unpack(%arg0: tensor<2x64x128xf32>, %arg1: tensor<2x128x64xf32>) -> tensor<2x64x64xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<2x8x128x8x1xf32>
    %pack = linalg.pack %arg0 outer_dims_perm = [0, 1, 2] inner_dims_pos = [1, 2] inner_tiles = [8, 1] into %0 : tensor<2x64x128xf32> -> tensor<2x8x128x8x1xf32>
    %pack_0 = linalg.pack %arg1 outer_dims_perm = [0, 2, 1] inner_dims_pos = [2, 1] inner_tiles = [8, 1] into %0 : tensor<2x128x64xf32> -> tensor<2x8x128x8x1xf32>
    %1 = tensor.empty() : tensor<2x8x8x8x8xf32>
    %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<2x8x8x8x8xf32>) -> tensor<2x8x8x8x8xf32>
    %3 = linalg.batch_mmt4d ins(%pack, %pack_0 : tensor<2x8x128x8x1xf32>, tensor<2x8x128x8x1xf32>) outs(%2 : tensor<2x8x8x8x8xf32>) -> tensor<2x8x8x8x8xf32>
    %4 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%3 : tensor<2x8x8x8x8xf32>) outs(%1 : tensor<2x8x8x8x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %6 = arith.maximumf %in, %cst : f32
      linalg.yield %6 : f32
    } -> tensor<2x8x8x8x8xf32>
    %5 = tensor.empty() : tensor<2x64x64xf32>
    %unpack = linalg.unpack %4 outer_dims_perm = [0, 1, 2] inner_dims_pos = [1, 2] inner_tiles = [8, 8] into %5 : tensor<2x8x8x8x8xf32> -> tensor<2x64x64xf32>
    util.return %unpack : tensor<2x64x64xf32>
  }

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
  // CHECK-LABEL: util.func public @transpose_epilogue
  // CHECK: %[[MM:.*]] = linalg.mmt4d
  // CHECK-NEXT: flow.return %[[MM]]
  // CHECK: flow.dispatch.region
  // CHECK: linalg.generic
  // CHECK: flow.return
  // CHECK: flow.dispatch.region -> (tensor<64x64xf32>)
  // CHECK: %[[UNPACK:.*]] = linalg.unpack
  // CHECK-NEXT: flow.return %[[UNPACK]]
  util.func public @transpose_epilogue(%arg0: tensor<64x128xf32>, %arg1: tensor<128x64xf32>) -> tensor<64x64xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<8x128x8x1xf32>
    %pack = linalg.pack %arg0 outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [8, 1] into %0 : tensor<64x128xf32> -> tensor<8x128x8x1xf32>
    %pack_0 = linalg.pack %arg1 outer_dims_perm = [1, 0] inner_dims_pos = [1, 0] inner_tiles = [8, 1] into %0 : tensor<128x64xf32> -> tensor<8x128x8x1xf32>
    %1 = tensor.empty() : tensor<8x8x8x8xf32>
    %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xf32>
    %3 = linalg.mmt4d ins(%pack, %pack_0 : tensor<8x128x8x1xf32>, tensor<8x128x8x1xf32>) outs(%2 : tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xf32>
    %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d1, d0, d2, d3)>, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3 : tensor<8x8x8x8xf32>) outs(%1 : tensor<8x8x8x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %6 = arith.maximumf %in, %cst : f32
      linalg.yield %6 : f32
    } -> tensor<8x8x8x8xf32>
    %5 = tensor.empty() : tensor<64x64xf32>
    %unpack = linalg.unpack %4 outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [8, 8] into %5 : tensor<8x8x8x8xf32> -> tensor<64x64xf32>
    util.return %unpack : tensor<64x64xf32>
  }

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
  // CHECK-LABEL: util.func public @multi_use
  // CHECK: %[[MM:.*]] = linalg.mmt4d
  // CHECK-NEXT: flow.return %[[MM]]
  // CHECK: flow.dispatch.region
  // CHECK: linalg.generic
  // CHECK: flow.return
  // CHECK: flow.dispatch.region -> (tensor<64x64xf32>)
  // CHECK: %[[UNPACK:.*]] = linalg.unpack
  // CHECK-NEXT: flow.return %[[UNPACK]]
  util.func public @multi_use(%arg0: tensor<64x128xf32>, %arg1: tensor<128x64xf32>) -> (tensor<64x64xf32>, tensor<8x8x8x8xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<8x128x8x1xf32>
    %pack = linalg.pack %arg0 outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [8, 1] into %0 : tensor<64x128xf32> -> tensor<8x128x8x1xf32>
    %pack_0 = linalg.pack %arg1 outer_dims_perm = [1, 0] inner_dims_pos = [1, 0] inner_tiles = [8, 1] into %0 : tensor<128x64xf32> -> tensor<8x128x8x1xf32>
    %1 = tensor.empty() : tensor<8x8x8x8xf32>
    %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xf32>
    %3 = linalg.mmt4d ins(%pack, %pack_0 : tensor<8x128x8x1xf32>, tensor<8x128x8x1xf32>) outs(%2 : tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xf32>
    %4 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3 : tensor<8x8x8x8xf32>) outs(%1 : tensor<8x8x8x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %6 = arith.maximumf %in, %cst : f32
      linalg.yield %6 : f32
    } -> tensor<8x8x8x8xf32>
    %5 = tensor.empty() : tensor<64x64xf32>
    %unpack = linalg.unpack %4 outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [8, 8] into %5 : tensor<8x8x8x8xf32> -> tensor<64x64xf32>
    util.return %unpack, %3 : tensor<64x64xf32>, tensor<8x8x8x8xf32>
  }

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
  // CHECK-LABEL: util.func public @truncation_epilogue
  // CHECK: %[[LHS:.*]] = flow.dispatch.region
  // CHECK: linalg.pack
  // CHECK: flow.return
  // CHECK: %[[RHS:.*]] = flow.dispatch.region
  // CHECK: linalg.pack
  // CHECK: flow.return
  // CHECK: %[[RESULT:.*]] = flow.dispatch.region -> (tensor<64x64xbf16>)
  // CHECK: %[[MM:.*]] = linalg.mmt4d ins(%[[LHS]], %[[RHS]]
  // CHECK-NOT: flow.return
  // CHECK: %[[EPILOGUE:.*]] = linalg.generic
  // CHECK-SAME: ins(%[[MM]] : tensor<8x8x8x8xf32>)
  // CHECK: arith.maximumf
  // CHECK-NOT: flow.return
  // CHECK: arith.truncf
  // CHECK-NOT: flow.return
  // CHECK: %[[UNPACK:.*]] = linalg.unpack
  // CHECK: flow.return %[[UNPACK]] : tensor<64x64xbf16>
  // CHECK-NOT: flow.dispatch.region
  // CHECK: util.return %[[RESULT]]
  util.func public @truncation_epilogue(%arg0: tensor<64x128xf32>, %arg1: tensor<128x64xf32>) -> tensor<64x64xbf16> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<8x128x8x1xf32>
    %pack = linalg.pack %arg0 outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [8, 1] into %0 : tensor<64x128xf32> -> tensor<8x128x8x1xf32>
    %pack_0 = linalg.pack %arg1 outer_dims_perm = [1, 0] inner_dims_pos = [1, 0] inner_tiles = [8, 1] into %0 : tensor<128x64xf32> -> tensor<8x128x8x1xf32>
    %1 = tensor.empty() : tensor<8x8x8x8xf32>
    %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xf32>
    %3 = linalg.mmt4d ins(%pack, %pack_0 : tensor<8x128x8x1xf32>, tensor<8x128x8x1xf32>) outs(%2 : tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xf32>
    %4 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3 : tensor<8x8x8x8xf32>) outs(%1 : tensor<8x8x8x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %6 = arith.maximumf %in, %cst : f32
      linalg.yield %6 : f32
    } -> tensor<8x8x8x8xf32>
    %cast_empty = tensor.empty() : tensor<8x8x8x8xbf16>
    %cast = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%4 : tensor<8x8x8x8xf32>) outs(%cast_empty : tensor<8x8x8x8xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %value = arith.truncf %in : f32 to bf16
      linalg.yield %value : bf16
    } -> tensor<8x8x8x8xbf16>
    %5 = tensor.empty() : tensor<64x64xbf16>
    %unpack = linalg.unpack %cast outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [8, 8] into %5 : tensor<8x8x8x8xbf16> -> tensor<64x64xbf16>
    util.return %unpack : tensor<64x64xbf16>
  }
