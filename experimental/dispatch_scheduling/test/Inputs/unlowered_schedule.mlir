module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %roots = transform.structured.match ops{["linalg.mmt4d"]} in %root
      : (!transform.any_op) -> !transform.any_op
    %regions, %operations = transform.iree.wrap_in_payload_region %roots
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
