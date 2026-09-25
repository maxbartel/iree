module attributes {transform.with_named_sequence} {
  transform.named_sequence @record(%executable: !transform.any_op {transform.readonly},
      %choice: !transform.any_param {transform.readonly}, %env: !transform.any_param {transform.readonly}) {
    %export = transform.structured.match ops{["flow.executable.export"]} in %executable
      : (!transform.any_op) -> !transform.any_op
    transform.annotate %export "iree_payload.schedule" = %choice
      : !transform.any_op, !transform.any_param
    transform.annotate %export "iree_payload.schedule_environment" = %env
      : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.consumed},
      %env: !transform.any_param {transform.readonly}) {
    %a = transform.param.constant {entry_point = "schedule_a", configuration = {tile = 1 : i64}} -> !transform.any_param
    %b = transform.param.constant {entry_point = "schedule_b", configuration = {tile = 1 : i64}} -> !transform.any_param
    %c = transform.param.constant {entry_point = "schedule_a", configuration = {tile = 2 : i64}} -> !transform.any_param
    %first = transform.structured.match ops{["flow.executable"]} attributes {sym_name = "first"} in %root
      : (!transform.any_op) -> !transform.any_op
    transform.include @record failures(propagate) (%first, %a, %env)
      : (!transform.any_op, !transform.any_param, !transform.any_param) -> ()
    %different_entry = transform.structured.match ops{["flow.executable"]} attributes {sym_name = "different_entry"} in %root
      : (!transform.any_op) -> !transform.any_op
    transform.include @record failures(propagate) (%different_entry, %b, %env)
      : (!transform.any_op, !transform.any_param, !transform.any_param) -> ()
    %different_config = transform.structured.match ops{["flow.executable"]} attributes {sym_name = "different_config"} in %root
      : (!transform.any_op) -> !transform.any_op
    transform.include @record failures(propagate) (%different_config, %c, %env)
      : (!transform.any_op, !transform.any_param, !transform.any_param) -> ()
    %duplicate = transform.structured.match ops{["flow.executable"]} attributes {sym_name = "duplicate"} in %root
      : (!transform.any_op) -> !transform.any_op
    transform.include @record failures(propagate) (%duplicate, %a, %env)
      : (!transform.any_op, !transform.any_param, !transform.any_param) -> ()
    %deduplicated = transform.apply_registered_pass "iree-flow-deduplicate-executables" to %root
      : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
  transform.named_sequence @schedule_a(%executable: !transform.any_op {transform.readonly},
      %config: !transform.any_param {transform.readonly}) {
    %payload = transform.structured.match ops{["iree_payload.region"]} in %executable
      : (!transform.any_op) -> !transform.any_op
    transform.annotate %payload "test.executed_a" = %config
      : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @schedule_b(%executable: !transform.any_op {transform.readonly},
      %config: !transform.any_param {transform.readonly}) {
    %payload = transform.structured.match ops{["iree_payload.region"]} in %executable
      : (!transform.any_op) -> !transform.any_op
    transform.annotate %payload "test.executed_b" = %config
      : !transform.any_op, !transform.any_param
    transform.yield
  }
}
