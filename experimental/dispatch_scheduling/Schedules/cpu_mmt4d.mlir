// Copyright 2026 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// The matcher is read-only. Once selected, every failure propagates.
// First form all selected tensor dispatches. Outline and deduplicate them before
// running the numerical schedule once per unique body. Unmatched computation
// stays in the global graph for normal dispatch creation.
module attributes {transform.with_named_sequence} {
  transform.named_sequence @match_input0(
      %function: !transform.any_op {transform.readonly}, %environment: !transform.any_param {transform.readonly})
      -> (!transform.any_op, !transform.any_op, !transform.any_param) {
    %roots = transform.iree.cpu.match_mmt4d_fusion_roots %function <{contraction_input = 0}>
      : (!transform.any_op) -> !transform.any_op
    transform.yield %function, %roots, %environment
      : !transform.any_op, !transform.any_op, !transform.any_param
  }
  transform.named_sequence @form_input0(
      %function: !transform.any_op {transform.readonly},
      %roots: !transform.any_op {transform.readonly}, %environment: !transform.any_param {transform.readonly}) {
    %retied_roots = transform.foreach %roots : !transform.any_op -> !transform.any_op {
    ^bb0(%mm: !transform.any_op):
      %epilogue = transform.get_consumers_of_result %mm[0]
        : (!transform.any_op) -> !transform.any_op
      %reused = transform.iree.reuse_linalg_input_as_init %epilogue[0]
        : (!transform.any_op) -> !transform.any_op
      transform.yield %mm : !transform.any_op
    }
    transform.apply_patterns to %function {
      transform.apply_patterns.linalg.erase_unnecessary_inputs
    } : !transform.any_op
    transform.foreach %retied_roots : !transform.any_op {
    ^bb0(%mm: !transform.any_op):
      transform.include @form_group failures(propagate) (%mm, %environment)
        : (!transform.any_op, !transform.any_param) -> ()
      transform.yield
    }
    transform.apply_patterns to %function {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %function : !transform.any_op
    transform.yield
  }
  transform.named_sequence @match_input1(
      %function: !transform.any_op {transform.readonly}, %environment: !transform.any_param {transform.readonly})
      -> (!transform.any_op, !transform.any_op, !transform.any_param) {
    %roots = transform.iree.cpu.match_mmt4d_fusion_roots %function <{contraction_input = 1}>
      : (!transform.any_op) -> !transform.any_op
    transform.yield %function, %roots, %environment
      : !transform.any_op, !transform.any_op, !transform.any_param
  }
  transform.named_sequence @form_input1(
      %function: !transform.any_op {transform.readonly},
      %roots: !transform.any_op {transform.readonly}, %environment: !transform.any_param {transform.readonly}) {
    %retied_roots = transform.foreach %roots : !transform.any_op -> !transform.any_op {
    ^bb0(%mm: !transform.any_op):
      %epilogue = transform.get_consumers_of_result %mm[0]
        : (!transform.any_op) -> !transform.any_op
      %reused = transform.iree.reuse_linalg_input_as_init %epilogue[1]
        : (!transform.any_op) -> !transform.any_op
      transform.yield %mm : !transform.any_op
    }
    transform.apply_patterns to %function {
      transform.apply_patterns.linalg.erase_unnecessary_inputs
    } : !transform.any_op
    transform.foreach %retied_roots : !transform.any_op {
    ^bb0(%mm: !transform.any_op):
      transform.include @form_group failures(propagate) (%mm, %environment)
        : (!transform.any_op, !transform.any_param) -> ()
      transform.yield
    }
    transform.apply_patterns to %function {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %function : !transform.any_op
    transform.yield
  }
  transform.named_sequence @form_group(
      %mm: !transform.any_op {transform.consumed}, %environment: !transform.any_param {transform.readonly}) {
    %m, %n = transform.iree.cpu.get_mmt4d_workgroup_tiles %mm
      : (!transform.any_op) -> (!transform.param<i64>, !transform.param<i64>)
    %init = transform.get_producer_of_operand %mm[2]
      : (!transform.any_op) -> !transform.any_op
    %fresh_init = transform.iree.replace_linalg_init_with_empty %init
      : (!transform.any_op) -> !transform.any_op
    %retied = transform.get_consumers_of_result %mm[0]
      : (!transform.any_op) -> !transform.any_op
    %tiled_epilogue, %grid = transform.structured.tile_using_forall %retied
      tile_sizes [%m, %n]
      : (!transform.any_op, !transform.param<i64>, !transform.param<i64>) -> (!transform.any_op, !transform.any_op)
    %tiled_mm, %grid1 = transform.structured.fuse_into_containing_op %mm into %grid
      : (!transform.any_op, !transform.any_op)
        -> (!transform.any_op, !transform.any_op)
    %tiled_init, %grid2 = transform.structured.fuse_into_containing_op
      %fresh_init into %grid1
      : (!transform.any_op, !transform.any_op)
        -> (!transform.any_op, !transform.any_op)

    %workgroups = transform.iree.forall_to_flow %grid2
      : (!transform.any_op) -> !transform.any_op

    %group = transform.structured.match
      ops{["linalg.fill", "linalg.mmt4d", "linalg.generic"]} in %workgroups
      : (!transform.any_op) -> !transform.any_op
    %payload = transform.iree.wrap_in_payload_group %group
      : (!transform.any_op) -> !transform.any_op

    // The choice is part of dispatch equivalence before deduplication. The
    // environment is an immutable library digest and resolved device target
    // supplied by the driver. Configuration belongs to the chosen script.
    %choice = transform.param.constant
      {entry_point = "cpu_mmt4d", configuration = {}} -> !transform.any_param
    transform.annotate %workgroups "iree_payload.schedule" = %choice
      : !transform.any_op, !transform.any_param
    transform.annotate %workgroups "iree_payload.schedule_environment" = %environment
      : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @compile_payload(%payload: !transform.any_op {transform.consumed}) {
    // Outlining changed the surrounding alias-analysis scope from workgroups
    // to its entry function. The computation is still entirely tensor-valued.
    %scope = transform.get_parent_op %payload <{op_name = "func.func"}>
      : (!transform.any_op) -> !transform.any_op
    %packed_mm = transform.structured.match ops{["linalg.mmt4d"]} in %payload
      : (!transform.any_op) -> !transform.any_op
    %add = transform.get_consumers_of_result %packed_mm[0]
      : (!transform.any_op) -> !transform.any_op

    %add_tile, %loops:2 = transform.structured.fuse %add
      tile_sizes [1, 1, 0, 0] apply_cleanup
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.apply_patterns to %payload {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    %uk_payload = transform.apply_registered_pass "iree-codegen-cpu-lower-to-ukernels"
      to %payload : (!transform.any_op) -> !transform.any_op
    %uk_epilogue = transform.structured.match ops{["linalg.generic"]} in %uk_payload
      : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %uk_epilogue : !transform.any_op
    transform.apply_patterns to %uk_payload {
      transform.apply_patterns.tensor.fold_tensor_subset_ops_into_vector_transfers
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    %with_abi = transform.iree.bufferize_payload_boundaries %scope <{bufferize_body}>
      : (!transform.any_op) -> !transform.any_op
    %region = transform.structured.match ops{["iree_payload.region"]}
      in %with_abi : (!transform.any_op) -> !transform.any_op
    %compute = transform.iree.outline_payload_region %region
      : (!transform.any_op) -> !transform.any_op

    transform.apply_patterns to %compute {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    %prepared = transform.apply_registered_pass "iree-llvmcpu-prepare-payload-codegen"
      to %compute : (!transform.any_op) -> !transform.any_op

    %calls = transform.apply_registered_pass "iree-codegen-lower-ukernel-ops-to-calls"
      to %prepared : (!transform.any_op) -> !transform.any_op
    %body = transform.structured.match ops{["func.func"]} in %calls
      : (!transform.any_op) -> !transform.any_op

    transform.apply_patterns to %body {
      transform.apply_patterns.vector.lower_contraction
        lowering_strategy = "outerproduct"
      transform.apply_patterns.vector.transfer_permutation_patterns
      transform.apply_patterns.vector.lower_transpose
        lowering_strategy = "shuffle_1d"
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_patterns to %body {
      transform.apply_patterns.vector.transfer_to_scf
        max_transfer_rank = 1 full_unroll = true
      transform.apply_patterns.vector.lower_transfer max_transfer_rank = 1
      transform.apply_patterns.canonicalization
    } : !transform.any_op

    %metadata = transform.apply_registered_pass "expand-strided-metadata"
      to %calls : (!transform.any_op) -> !transform.any_op
    %affine = transform.apply_registered_pass "lower-affine" to %metadata
      : (!transform.any_op) -> !transform.any_op
    %cfg = transform.apply_registered_pass "convert-scf-to-cf" to %affine
      : (!transform.any_op) -> !transform.any_op
    %conversion0 = transform.apply_registered_pass "convert-vector-to-llvm" to %cfg
      : (!transform.any_op) -> !transform.any_op
    %conversion1 = transform.apply_registered_pass "finalize-memref-to-llvm" to %conversion0
      : (!transform.any_op) -> !transform.any_op
    %conversion2 = transform.apply_registered_pass "convert-math-to-llvm" to %conversion1
      : (!transform.any_op) -> !transform.any_op
    %conversion3 = transform.apply_registered_pass "convert-arith-to-llvm" to %conversion2
      : (!transform.any_op) -> !transform.any_op
    %conversion4 = transform.apply_registered_pass "convert-index-to-llvm" to %conversion3
      : (!transform.any_op) -> !transform.any_op
    %conversion5 = transform.apply_registered_pass "convert-cf-to-llvm" to %conversion4
      : (!transform.any_op) -> !transform.any_op
    %llvm = transform.apply_registered_pass "convert-func-to-llvm" to %conversion5
      : (!transform.any_op) -> !transform.any_op
    %ub = transform.apply_registered_pass "convert-ub-to-llvm" to %llvm
      : (!transform.any_op) -> !transform.any_op
    %finished = transform.apply_registered_pass "reconcile-unrealized-casts"
      to %ub : (!transform.any_op) -> !transform.any_op

    %verified = transform.apply_registered_pass "iree-llvmcpu-verify-payload-codegen"
      to %finished : (!transform.any_op) -> !transform.any_op

    transform.yield
  }
  // Parallel entry point. Its handle is one isolated executable, never the
  // global graph. All numerical transformations remain in this library.
  transform.named_sequence @cpu_mmt4d(%executable: !transform.any_op {transform.readonly}) {
    %payload = transform.structured.match ops{["iree_payload.region"]} in %executable
      : (!transform.any_op) -> !transform.any_op
    transform.include @compile_payload failures(propagate) (%payload)
      : (!transform.any_op) -> ()
    transform.yield
  }
  transform.named_sequence @__transform_main(
      %root: !transform.any_op {transform.consumed}, %environment: !transform.any_param {transform.readonly}) {
    %after0 = transform.foreach_match in %root, %environment @match_input0 -> @form_input0
      : (!transform.any_op, !transform.any_param) -> !transform.any_op
    %after1 = transform.foreach_match in %after0, %environment @match_input1 -> @form_input1
      : (!transform.any_op, !transform.any_param) -> !transform.any_op
    // Capture dynamic binding dimensions before outlining, as in normal Flow.
    // Later host passes no longer visit these outlined entry functions.
    %shaped = transform.apply_registered_pass "iree-flow-capture-dynamic-dims"
      to %after1 : (!transform.any_op) -> !transform.any_op

    // No numerical lowering has happened yet. Structural equivalence includes
    // the workgroup-count region, tensor bindings and complete fused body.
    %outlined = transform.apply_registered_pass "iree-flow-outline-dispatch-regions"
      to %shaped : (!transform.any_op) -> !transform.any_op
    %unique = transform.apply_registered_pass "iree-flow-deduplicate-executables"
      to %outlined : (!transform.any_op) -> !transform.any_op
    // The global stage ends here. The driver runs the recorded named sequence
    // once per unique executable, in parallel with independent interpreter
    // state. There is no numerical C++ codegen pipeline behind that driver.
    transform.yield
  }
}
