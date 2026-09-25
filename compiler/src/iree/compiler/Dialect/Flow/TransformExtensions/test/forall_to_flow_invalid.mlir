// RUN: iree-opt %s --transform-interpreter --verify-diagnostics --split-input-file

func.func @nonnormalized() {
  // expected-error @+2 {{expected a normalized forall with one to three dimensions}}
  // expected-note @+1 {{attempted to apply to this op}}
  scf.forall (%i) = (1) to (4) step (1) {
    scf.forall.in_parallel { }
  }
  return
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %loops = transform.structured.match ops{["scf.forall"]} in %root : (!transform.any_op) -> !transform.any_op
    // expected-error @+1 {{failed to apply}}
    %dispatches = transform.iree.forall_to_flow %loops : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

func.func @rank_four() {
  // expected-error @+2 {{expected a normalized forall with one to three dimensions}}
  // expected-note @+1 {{attempted to apply to this op}}
  scf.forall (%i, %j, %k, %l) in (2, 2, 2, 2) {
    scf.forall.in_parallel { }
  }
  return
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %loops = transform.structured.match ops{["scf.forall"]} in %root : (!transform.any_op) -> !transform.any_op
    // expected-error @+1 {{failed to apply}}
    %dispatches = transform.iree.forall_to_flow %loops : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

func.func @shared_alias(%init: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  // expected-error @+2 {{expected distinct shared output operands}}
  // expected-note @+1 {{attempted to apply to this op}}
  %r:2 = scf.forall (%i) in (4) shared_outs(%a = %init, %b = %init) -> (tensor<4xf32>, tensor<4xf32>) {
    %sa = tensor.extract_slice %a[%i] [1] [1] : tensor<4xf32> to tensor<1xf32>
    %sb = tensor.extract_slice %b[%i] [1] [1] : tensor<4xf32> to tensor<1xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %sa into %a[%i] [1] [1] : tensor<1xf32> into tensor<4xf32>
      tensor.parallel_insert_slice %sb into %b[%i] [1] [1] : tensor<1xf32> into tensor<4xf32>
    }
  }
  return %r#0, %r#1 : tensor<4xf32>, tensor<4xf32>
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %loops = transform.structured.match ops{["scf.forall"]} in %root : (!transform.any_op) -> !transform.any_op
    // expected-error @+1 {{failed to apply}}
    %dispatches = transform.iree.forall_to_flow %loops : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
