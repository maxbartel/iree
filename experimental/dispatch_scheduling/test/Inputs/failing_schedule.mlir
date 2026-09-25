module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.consumed}) {
    %changed = transform.apply_registered_pass "does-not-exist" to %root
      : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
