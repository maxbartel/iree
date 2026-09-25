// RUN: iree-opt --iree-plugin=dispatch_scheduling %s --split-input-file --transform-interpreter --verify-diagnostics

func.func @read_init(%input: tensor<4x8xf32>, %old: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %r = linalg.generic {
    indexing_maps = [affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i,j)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%input : tensor<4x8xf32>) outs(%old : tensor<4x8xf32>) {
  ^bb0(%x: f32, %unused: f32):
    linalg.yield %unused : f32
  } -> tensor<4x8xf32>
  return %r : tensor<4x8xf32>
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %generic = transform.structured.match ops{["linalg.generic"]} in %root
      : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{destination block argument must be unused}}
    %retied = transform.iree.reuse_linalg_input_as_init %generic[0]
      : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

func.func @different_maps(%input: tensor<8x8xf32>, %old: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %r = linalg.generic {
    indexing_maps = [affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(j,i)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%input : tensor<8x8xf32>) outs(%old : tensor<8x8xf32>) {
  ^bb0(%x: f32, %unused: f32):
    linalg.yield %x : f32
  } -> tensor<8x8xf32>
  return %r : tensor<8x8xf32>
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %generic = transform.structured.match ops{["linalg.generic"]} in %root
      : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{expected identical permutation input and output indexing maps}}
    %retied = transform.iree.reuse_linalg_input_as_init %generic[0]
      : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

func.func @dynamic(%input: tensor<?x8xf32>, %old: tensor<?x8xf32>) -> tensor<?x8xf32> {
  %r = linalg.generic {
    indexing_maps = [affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i,j)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%input : tensor<?x8xf32>) outs(%old : tensor<?x8xf32>) {
  ^bb0(%x: f32, %unused: f32):
    linalg.yield %x : f32
  } -> tensor<?x8xf32>
  return %r : tensor<?x8xf32>
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %generic = transform.structured.match ops{["linalg.generic"]} in %root
      : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{input and destination shapes must be provably equal}}
    %retied = transform.iree.reuse_linalg_input_as_init %generic[0]
      : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

func.func @input_out_of_range(%input: tensor<4x8xf32>, %old: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %r = linalg.generic {
    indexing_maps = [affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i,j)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%input : tensor<4x8xf32>) outs(%old : tensor<4x8xf32>) {
  ^bb0(%x: f32, %unused: f32):
    linalg.yield %x : f32
  } -> tensor<4x8xf32>
  return %r : tensor<4x8xf32>
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %generic = transform.structured.match ops{["linalg.generic"]} in %root
      : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{DPS input or output index out of range}}
    %retied = transform.iree.reuse_linalg_input_as_init %generic[1]
      : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

func.func @negative_input(%input: tensor<4x8xf32>, %old: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %r = linalg.generic {
    indexing_maps = [affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i,j)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%input : tensor<4x8xf32>) outs(%old : tensor<4x8xf32>) {
  ^bb0(%x: f32, %unused: f32):
    linalg.yield %x : f32
  } -> tensor<4x8xf32>
  return %r : tensor<4x8xf32>
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %generic = transform.structured.match ops{["linalg.generic"]} in %root
      : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{DPS input or output index out of range}}
    %retied = transform.iree.reuse_linalg_input_as_init %generic[-1]
      : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

func.func @output_out_of_range(%input: tensor<4x8xf32>, %old: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %r = linalg.generic {
    indexing_maps = [affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i,j)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%input : tensor<4x8xf32>) outs(%old : tensor<4x8xf32>) {
  ^bb0(%x: f32, %unused: f32):
    linalg.yield %x : f32
  } -> tensor<4x8xf32>
  return %r : tensor<4x8xf32>
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %generic = transform.structured.match ops{["linalg.generic"]} in %root
      : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{DPS input or output index out of range}}
    %retied = transform.iree.reuse_linalg_input_as_init %generic[0] <{output = 1 : i64}>
      : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

func.func @type_mismatch(%input: tensor<8x4xf32>, %old: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %r = linalg.generic {
    indexing_maps = [affine_map<(i,j)->(j,i)>, affine_map<(i,j)->(i,j)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%input : tensor<8x4xf32>) outs(%old : tensor<4x8xf32>) {
  ^bb0(%x: f32, %unused: f32):
    linalg.yield %x : f32
  } -> tensor<4x8xf32>
  return %r : tensor<4x8xf32>
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %generic = transform.structured.match ops{["linalg.generic"]} in %root
      : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{expected matching tensor types}}
    %retied = transform.iree.reuse_linalg_input_as_init %generic[0]
      : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

func.func @reduction(%input: tensor<4x8xf32>, %old: tensor<4xf32>) -> tensor<4xf32> {
  %r = linalg.generic {
    indexing_maps = [affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i)>],
    iterator_types = ["parallel", "reduction"]}
    ins(%input : tensor<4x8xf32>) outs(%old : tensor<4xf32>) {
  ^bb0(%x: f32, %unused: f32):
    linalg.yield %x : f32
  } -> tensor<4xf32>
  return %r : tensor<4xf32>
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %generic = transform.structured.match ops{["linalg.generic"]} in %root
      : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{expected a pure tensor generic with only parallel iterators}}
    %retied = transform.iree.reuse_linalg_input_as_init %generic[0]
      : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
