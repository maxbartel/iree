// RUN: iree-opt --iree-plugin=dispatch_scheduling %s --split-input-file --transform-interpreter --verify-diagnostics
func.func @buffer(%out: memref<4xf32>, %v: f32) {
  linalg.fill ins(%v : f32) outs(%out : memref<4xf32>)
  return
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %ops = transform.structured.match ops{["linalg.fill", "linalg.generic"]} in %root : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{expected distinct, pure tensor DPS operations in one block}}
    %payload = transform.iree.wrap_in_payload_group %ops : (!transform.any_op) -> !transform.any_op
    %none = transform.structured.match ops{["test.absent"]} in %root : (!transform.any_op) -> !transform.any_op
    %empty = transform.iree.wrap_in_payload_group %none : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

func.func @chain_and_old_input(%out: tensor<4xf32>, %factor: f32) -> (tensor<4xf32>, tensor<4xf32>) {
  %fill = linalg.fill ins(%factor : f32) outs(%out : tensor<4xf32>) -> tensor<4xf32>
  %r = linalg.generic {indexing_maps = [affine_map<(i)->(i)>, affine_map<(i)->(i)>, affine_map<(i)->(i)>], iterator_types = ["parallel"]} ins(%out, %fill : tensor<4xf32>, tensor<4xf32>) outs(%fill : tensor<4xf32>) {
  ^bb0(%old: f32, %value: f32, %unused: f32):
    %sum = arith.addf %old, %value : f32
    linalg.yield %sum : f32
  } -> tensor<4xf32>
  return %fill, %r : tensor<4xf32>, tensor<4xf32>
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %ops = transform.structured.match ops{["linalg.fill", "linalg.generic"]} in %root : (!transform.any_op) -> !transform.any_op
    // Two live results cannot tie to the same external destination.
    // expected-error @below {{payload group requires distinct external destinations}}
    %payload = transform.iree.wrap_in_payload_group %ops : (!transform.any_op) -> !transform.any_op
    %none = transform.structured.match ops{["test.absent"]} in %root : (!transform.any_op) -> !transform.any_op
    %empty = transform.iree.wrap_in_payload_group %none : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

func.func @first(%out: tensor<4xf32>, %factor: f32) -> tensor<4xf32> {
  %fill = linalg.fill ins(%factor : f32) outs(%out : tensor<4xf32>) -> tensor<4xf32>
  %r = linalg.generic {indexing_maps = [affine_map<(i)->(i)>, affine_map<(i)->(i)>, affine_map<(i)->(i)>], iterator_types = ["parallel"]} ins(%out, %fill : tensor<4xf32>, tensor<4xf32>) outs(%fill : tensor<4xf32>) {
  ^bb0(%old: f32, %value: f32, %unused: f32):
    %sum = arith.addf %old, %value : f32
    linalg.yield %sum : f32
  } -> tensor<4xf32>
  return %r : tensor<4xf32>
}
func.func @second(%out: tensor<4xf32>, %factor: f32) -> tensor<4xf32> {
  %fill = linalg.fill ins(%factor : f32) outs(%out : tensor<4xf32>) -> tensor<4xf32>
  %r = linalg.generic {indexing_maps = [affine_map<(i)->(i)>, affine_map<(i)->(i)>, affine_map<(i)->(i)>], iterator_types = ["parallel"]} ins(%out, %fill : tensor<4xf32>, tensor<4xf32>) outs(%fill : tensor<4xf32>) {
  ^bb0(%old: f32, %value: f32, %unused: f32):
    %sum = arith.addf %old, %value : f32
    linalg.yield %sum : f32
  } -> tensor<4xf32>
  return %r : tensor<4xf32>
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %ops = transform.structured.match ops{["linalg.fill", "linalg.generic"]} in %root : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{expected distinct, pure tensor DPS operations in one block}}
    %payload = transform.iree.wrap_in_payload_group %ops : (!transform.any_op) -> !transform.any_op
    %none = transform.structured.match ops{["test.absent"]} in %root : (!transform.any_op) -> !transform.any_op
    %empty = transform.iree.wrap_in_payload_group %none : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
// -----
func.func @repeated_operation(%out: tensor<4xf32>, %v: f32) -> tensor<4xf32> {
  // expected-note @below {{repeated target op}}
  %r = linalg.fill ins(%v : f32) outs(%out : tensor<4xf32>) -> tensor<4xf32>
  return %r : tensor<4xf32>
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %fill = transform.structured.match ops{["linalg.fill"]} in %root : (!transform.any_op) -> !transform.any_op
    %ops = transform.merge_handles %fill, %fill : !transform.any_op
    // expected-error @below {{points to a payload entity more than once}}
    %payload = transform.iree.wrap_in_payload_group %ops : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----
func.func @repeated_destination(%out: tensor<4xf32>, %v: f32) -> (tensor<4xf32>, tensor<4xf32>) {
  %f = linalg.fill ins(%v : f32) outs(%out : tensor<4xf32>) -> tensor<4xf32>
  %r:2 = linalg.generic {indexing_maps = [affine_map<(i)->(i)>, affine_map<(i)->(i)>], iterator_types = ["parallel"]} outs(%f, %f : tensor<4xf32>, tensor<4xf32>) {
  ^bb0(%x: f32, %y: f32):
    linalg.yield %x, %y : f32, f32
  } -> (tensor<4xf32>, tensor<4xf32>)
  return %r#0, %r#1 : tensor<4xf32>, tensor<4xf32>
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %ops = transform.structured.match ops{["linalg.fill", "linalg.generic"]} in %root : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{payload group requires distinct external destinations}}
    %payload = transform.iree.wrap_in_payload_group %ops : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// Moving the producer into a payload at the last selected operation would
// create a cycle through the unselected scalar read.
func.func @external_user_before_root(%a: tensor<4xf32>, %b: tensor<4xf32>, %v: f32) -> (tensor<4xf32>, tensor<4xf32>) {
  %c0 = arith.constant 0 : index
  %first = linalg.fill ins(%v : f32) outs(%a : tensor<4xf32>) -> tensor<4xf32>
  %read = tensor.extract %first[%c0] : tensor<4xf32>
  %second = linalg.fill ins(%read : f32) outs(%b : tensor<4xf32>) -> tensor<4xf32>
  return %first, %second : tensor<4xf32>, tensor<4xf32>
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %ops = transform.structured.match ops{["linalg.fill"]} in %root : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{external group result users must follow the last selected operation}}
    %payload = transform.iree.wrap_in_payload_group %ops : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
