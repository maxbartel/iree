// RUN: iree-opt --iree-plugin=dispatch_scheduling %s --split-input-file --transform-interpreter --verify-diagnostics
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.consumed}) {
    // expected-error @below {{expected an isolated, non-symbol-table scope enclosing payloads}}
    %result = transform.iree.bufferize_payload_boundaries %root : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----
func.func @nested(%out: memref<4xf32>) {
  iree_payload.region outs(%out : memref<4xf32>) {
  ^bb0(%a: memref<4xf32>):
    iree_payload.region outs(%a : memref<4xf32>) {
    ^bb0(%b: memref<4xf32>):
      iree_payload.yield
    }
    iree_payload.yield
  }
  return
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %scope = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{nested payloads and symbol tables are unsupported}}
    %result = transform.iree.bufferize_payload_boundaries %scope : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----
func.func @owned(%out: memref<4xf32>) {
  iree_payload.region outs(%out : memref<4xf32>) {
  ^bb0(%a: memref<4xf32>):
    builtin.module @compute {}
    iree_payload.yield
  }
  return
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %scope = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{nested payloads and symbol tables are unsupported}}
    %result = transform.iree.bufferize_payload_boundaries %scope : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
