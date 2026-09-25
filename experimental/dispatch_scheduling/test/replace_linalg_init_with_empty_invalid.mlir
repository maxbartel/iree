// RUN: iree-opt --iree-plugin=dispatch_scheduling %s --split-input-file --transform-interpreter --verify-diagnostics

func.func @read_old(%old: tensor<4xf32>) -> tensor<4xf32> {
  %r = linalg.generic {indexing_maps = [affine_map<(i)->(i)>], iterator_types = ["parallel"]}
    outs(%old : tensor<4xf32>) {
  ^bb0(%x: f32):
    %one = arith.constant 1.0 : f32
    %y = arith.addf %x, %one : f32
    linalg.yield %y : f32
  } -> tensor<4xf32>
  return %r : tensor<4xf32>
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %op = transform.structured.match ops{["linalg.generic"]} in %root
      : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{destination block argument must be unused}}
    %fresh = transform.iree.replace_linalg_init_with_empty %op
      : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

func.func @zero_reduction(%input: tensor<4x0xf32>, %old: tensor<4xf32>) -> tensor<4xf32> {
  %r = linalg.generic {indexing_maps = [affine_map<(i,k)->(i,k)>, affine_map<(i,k)->(i)>], iterator_types = ["parallel", "reduction"]}
    ins(%input : tensor<4x0xf32>) outs(%old : tensor<4xf32>) {
  ^bb0(%x: f32, %unused: f32):
    linalg.yield %x : f32
  } -> tensor<4xf32>
  return %r : tensor<4xf32>
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %op = transform.structured.match ops{["linalg.generic"]} in %root
      : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{expected a pure tensor Linalg op with only parallel iterators}}
    %fresh = transform.iree.replace_linalg_init_with_empty %op
      : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

func.func @partial_write(%input: tensor<2xf32>, %old: tensor<4xf32>) -> tensor<4xf32> {
  %r = linalg.generic {indexing_maps = [affine_map<(i)->(i)>, affine_map<(i)->(i * 2)>], iterator_types = ["parallel"]}
    ins(%input : tensor<2xf32>) outs(%old : tensor<4xf32>) {
  ^bb0(%x: f32, %unused: f32):
    linalg.yield %x : f32
  } -> tensor<4xf32>
  return %r : tensor<4xf32>
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %op = transform.structured.match ops{["linalg.generic"]} in %root
      : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{expected a permutation output map}}
    %fresh = transform.iree.replace_linalg_init_with_empty %op
      : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
