// RUN: iree-opt --transform-interpreter --split-input-file %s | FileCheck %s

// A rank-reducing slice captured by the forall is cloned into the dispatch.
// Its load must preserve the reduced result type, including for consumers
// that take another rank-reducing slice.
// CHECK-LABEL: func.func @rank_reducing_capture(
// CHECK: %[[PLANE:.*]] = iree_tensor_ext.dispatch.tensor.load %{{.*}}, offsets = [0, 1, 0], sizes = [4, 1, 16], strides = [1, 1, 1]
// CHECK-SAME: -> tensor<4x16xf32>
// CHECK: tensor.extract_slice %[[PLANE]][%{{.*}}, 0] [1, 16] [1, 1]
// CHECK-SAME: tensor<4x16xf32> to tensor<16xf32>
// CHECK: iree_tensor_ext.dispatch.tensor.load
// CHECK-SAME: sizes = [1, 16], strides = [1, 1]
// CHECK-SAME: -> tensor<16xf32>
// CHECK: linalg.copy
// CHECK-SAME: ins(%{{.*}} : tensor<16xf32>) outs(%{{.*}} : tensor<16xf32>)
func.func @rank_reducing_capture(%input: tensor<4x2x16xf32>,
                                %init: tensor<4x16xf32>) -> tensor<4x16xf32> {
  %plane = tensor.extract_slice %input[0, 1, 0] [4, 1, 16] [1, 1, 1]
    : tensor<4x2x16xf32> to tensor<4x16xf32>
  %r = scf.forall (%i) in (4) shared_outs(%out = %init) -> tensor<4x16xf32> {
    %row = tensor.extract_slice %plane[%i, 0] [1, 16] [1, 1]
      : tensor<4x16xf32> to tensor<16xf32>
    %dst = tensor.extract_slice %out[%i, 0] [1, 16] [1, 1]
      : tensor<4x16xf32> to tensor<16xf32>
    %copy = linalg.copy ins(%row : tensor<16xf32>) outs(%dst : tensor<16xf32>)
      -> tensor<16xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %copy into %out[%i, 0] [1, 16] [1, 1]
        : tensor<16xf32> into tensor<4x16xf32>
    }
  }
  return %r : tensor<4x16xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %loop = transform.structured.match ops{["scf.forall"]} in %root
      : (!transform.any_op) -> !transform.any_op
    %dispatch = transform.iree.forall_to_flow %loop
      : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// Explicit fusion keeps an unselected gather outside the dispatch, while a
// rank-reducing boundary view is still folded into the dispatch tensor load.
// CHECK-LABEL: func.func @explicit_fusion(
// CHECK: %[[GATHER:.*]] = iree_linalg_ext.gather
// CHECK: flow.dispatch.workgroups{{.*}}(%[[GATHER]],
// CHECK-NOT: iree_linalg_ext.gather
// CHECK: iree_tensor_ext.dispatch.tensor.load {{.*}} offsets = [0, 1, 0], sizes = [4, 1, 16], strides = [1, 1, 1]
// CHECK-SAME: -> tensor<4x16xf32>
// CHECK-NOT: iree_linalg_ext.gather
// CHECK: arith.negf
// CHECK-NOT: iree_linalg_ext.gather
// CHECK: return
func.func @explicit_fusion(%input: tensor<8x2x16xf32>, %ids: tensor<4x1xi32>,
                           %init: tensor<4x16xf32>) -> tensor<4x16xf32> {
  %empty = tensor.empty() : tensor<4x2x16xf32>
  %gather = iree_linalg_ext.gather dimension_map = [0]
    ins(%input, %ids : tensor<8x2x16xf32>, tensor<4x1xi32>)
    outs(%empty : tensor<4x2x16xf32>) -> tensor<4x2x16xf32>
  %plane = tensor.extract_slice %gather[0, 1, 0] [4, 1, 16] [1, 1, 1]
    : tensor<4x2x16xf32> to tensor<4x16xf32>
  %result = scf.forall (%i) in (4) shared_outs(%out = %init) -> tensor<4x16xf32> {
    %row = tensor.extract_slice %plane[%i, 0] [1, 16] [1, 1]
      : tensor<4x16xf32> to tensor<16xf32>
    %dest = tensor.extract_slice %out[%i, 0] [1, 16] [1, 1]
      : tensor<4x16xf32> to tensor<16xf32>
    %negated = linalg.generic {
        indexing_maps = [affine_map<(i)->(i)>, affine_map<(i)->(i)>],
        iterator_types = ["parallel"]}
        ins(%row : tensor<16xf32>) outs(%dest : tensor<16xf32>) {
    ^bb0(%value: f32, %unused: f32):
      %neg = arith.negf %value : f32
      linalg.yield %neg : f32
    } -> tensor<16xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %negated into %out[%i, 0] [1, 16] [1, 1]
        : tensor<16xf32> into tensor<4x16xf32>
    }
  }
  return %result : tensor<4x16xf32>
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %loop = transform.structured.match ops{["scf.forall"]} in %root : (!transform.any_op) -> !transform.any_op
    %dispatch = transform.iree.forall_to_flow %loop <{clone_tensor_compute = false}> : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// Verifies that `transform.iree.forall_to_flow` produces a
// `flow.dispatch.workgroups` whose count region returns the workload values
// padded out to three dimensions, regardless of whether the forall's upper
// bounds are static or dynamic.

// CHECK-LABEL: func.func @forall_to_flow_constant
// CHECK-SAME:      %[[LHS:[A-Za-z0-9_]+]]: tensor<128x128xf32>, %[[RHS:[A-Za-z0-9_]+]]: tensor<128x128xf32>, %[[INIT:[A-Za-z0-9_]+]]: tensor<128x128xf32>
// CHECK-DAG:     %[[C4_0:.+]] = arith.constant 4 : index
// CHECK-DAG:     %[[C4_1:.+]] = arith.constant 4 : index
// CHECK:         %[[D:.+]] = flow.dispatch.workgroups[%[[C4_0]], %[[C4_1]]]
// CHECK:           iree_tensor_ext.dispatch.tensor.load
// CHECK:           linalg.matmul
// CHECK:           iree_tensor_ext.dispatch.tensor.store
// CHECK:           flow.return
// CHECK:         count(%[[CM:[A-Za-z0-9_]+]]: index, %[[CN:[A-Za-z0-9_]+]]: index)
// CHECK:           %[[ONE:.+]] = arith.constant 1 : index
// CHECK:           flow.return %[[CM]], %[[CN]], %[[ONE]] : index, index, index
// CHECK:         return %[[D]]
func.func @forall_to_flow_constant(%lhs: tensor<128x128xf32>,
                                        %rhs: tensor<128x128xf32>,
                                        %init: tensor<128x128xf32>) -> tensor<128x128xf32> {
  %r = scf.forall (%i, %j) in (4, 4) shared_outs(%out = %init) -> tensor<128x128xf32> {
    %m_off = affine.apply affine_map<(d0) -> (d0 * 32)>(%i)
    %n_off = affine.apply affine_map<(d0) -> (d0 * 32)>(%j)
    %ls = tensor.extract_slice %lhs[%m_off, 0] [32, 128] [1, 1]
        : tensor<128x128xf32> to tensor<32x128xf32>
    %rs = tensor.extract_slice %rhs[0, %n_off] [128, 32] [1, 1]
        : tensor<128x128xf32> to tensor<128x32xf32>
    %os = tensor.extract_slice %out[%m_off, %n_off] [32, 32] [1, 1]
        : tensor<128x128xf32> to tensor<32x32xf32>
    %mm = linalg.matmul
        ins(%ls, %rs : tensor<32x128xf32>, tensor<128x32xf32>)
        outs(%os : tensor<32x32xf32>) -> tensor<32x32xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %mm into %out[%m_off, %n_off] [32, 32] [1, 1]
          : tensor<32x32xf32> into tensor<128x128xf32>
    }
  }
  return %r : tensor<128x128xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %forall = transform.structured.match ops{["scf.forall"]} in %root
        : (!transform.any_op) -> !transform.any_op
    %dispatch = transform.iree.forall_to_flow %forall
        : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// Dynamic upper bounds become the dispatch workload, and the count region
// returns them padded with a constant `1` for the missing third axis.

// CHECK-LABEL: func.func @forall_to_flow_dynamic
// CHECK-SAME:      %[[M_TILES:[A-Za-z0-9_]+]]: index,
// CHECK-SAME:      %[[N_TILES:[A-Za-z0-9_]+]]: index,
// CHECK-SAME:      %[[INIT:[A-Za-z0-9_]+]]: tensor<?x?xf32>
// CHECK:         %[[D:.+]] = flow.dispatch.workgroups[%[[M_TILES]], %[[N_TILES]]]
// CHECK:           iree_tensor_ext.dispatch.tensor.load
// CHECK:           iree_tensor_ext.dispatch.tensor.store
// CHECK:           flow.return
// CHECK:         count(%[[CM:[A-Za-z0-9_]+]]: index, %[[CN:[A-Za-z0-9_]+]]: index)
// CHECK:           %[[ONE:.+]] = arith.constant 1 : index
// CHECK:           flow.return %[[CM]], %[[CN]], %[[ONE]] : index, index, index
// CHECK:         return %[[D]]
func.func @forall_to_flow_dynamic(%m_tiles: index, %n_tiles: index,
                                       %init: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %r = scf.forall (%i, %j) in (%m_tiles, %n_tiles) shared_outs(%out = %init) -> tensor<?x?xf32> {
    %slice = tensor.extract_slice %out[%i, %j] [1, 1] [1, 1]
        : tensor<?x?xf32> to tensor<1x1xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %slice into %out[%i, %j] [1, 1] [1, 1]
          : tensor<1x1xf32> into tensor<?x?xf32>
    }
  }
  return %r : tensor<?x?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %forall = transform.structured.match ops{["scf.forall"]} in %root
        : (!transform.any_op) -> !transform.any_op
    %dispatch = transform.iree.forall_to_flow %forall
        : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL: func.func @three_dimensional(
// CHECK-SAME: %[[X:[a-zA-Z0-9_]+]]: index, %[[Y:[a-zA-Z0-9_]+]]: index, %[[Z:[a-zA-Z0-9_]+]]: index
// CHECK: flow.dispatch.workgroups[%[[X]], %[[Y]], %[[Z]]]
// CHECK: count(%[[CX:[a-zA-Z0-9_]+]]: index, %[[CY:[a-zA-Z0-9_]+]]: index, %[[CZ:[a-zA-Z0-9_]+]]: index)
// CHECK-NEXT: flow.return %[[CX]], %[[CY]], %[[CZ]] : index, index, index
func.func @three_dimensional(%x: index, %y: index, %z: index) {
  scf.forall (%i, %j, %k) in (%x, %y, %z) {
    scf.forall.in_parallel { }
  }
  return
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %loops = transform.structured.match ops{["scf.forall"]} in %root : (!transform.any_op) -> !transform.any_op
    %dispatches = transform.iree.forall_to_flow %loops : (!transform.any_op) -> !transform.any_op
    // Consuming the empty match must safely return an empty handle.
    %none = transform.structured.match ops{["scf.forall"]} in %root : (!transform.any_op) -> !transform.any_op
    %empty = transform.iree.forall_to_flow %none : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// Independent empty initializers may become the same SSA value after CSE.
// They must still produce independent output bindings, interleaved here with
// an initialized output that must retain its operand tie. Stores deliberately
// occur in reverse result order.
// CHECK-LABEL: func.func @independent_empty_results(
// CHECK-SAME: %[[INIT:[a-zA-Z0-9_]+]]: tensor<?xf32>, %[[N:[a-zA-Z0-9_]+]]: index
// CHECK: %[[EMPTY:.*]] = tensor.empty(%[[N]])
// CHECK-NOT: tensor.empty
// CHECK: %[[RESULT:.*]]:3 = flow.dispatch.workgroups
// CHECK-SAME: -> (tensor<?xf16>{%{{.*}}}, %[[INIT]]{%{{.*}}}, tensor<?xf16>{%{{.*}}})
// CHECK-NEXT: %[[MIDDLE:[a-zA-Z0-9_]+]]: !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?xf32>>
// CHECK-SAME: %[[FIRST:[a-zA-Z0-9_]+]]: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xf16>>
// CHECK-SAME: %[[LAST:[a-zA-Z0-9_]+]]: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xf16>>
// CHECK: %[[FIRST_TILE:.*]] = iree_tensor_ext.dispatch.tensor.load %[[FIRST]],
// CHECK: %[[MIDDLE_TILE:.*]] = iree_tensor_ext.dispatch.tensor.load %[[MIDDLE]],
// CHECK: %[[LAST_TILE:.*]] = iree_tensor_ext.dispatch.tensor.load %[[LAST]],
// CHECK: %[[FIRST_VALUE:.*]] = linalg.fill {{.*}} outs(%[[FIRST_TILE]] :
// CHECK: %[[MIDDLE_VALUE:.*]] = linalg.fill {{.*}} outs(%[[MIDDLE_TILE]] :
// CHECK: %[[LAST_VALUE:.*]] = linalg.fill {{.*}} outs(%[[LAST_TILE]] :
// CHECK: iree_tensor_ext.dispatch.tensor.store %[[LAST_VALUE]], %[[LAST]],
// CHECK: iree_tensor_ext.dispatch.tensor.store %[[MIDDLE_VALUE]], %[[MIDDLE]],
// CHECK: iree_tensor_ext.dispatch.tensor.store %[[FIRST_VALUE]], %[[FIRST]],
// CHECK: return %[[RESULT]]#0, %[[RESULT]]#1, %[[RESULT]]#2
func.func @independent_empty_results(%init: tensor<?xf32>, %n: index)
    -> (tensor<?xf16>, tensor<?xf32>, tensor<?xf16>) {
  %c0 = arith.constant 0 : index
  %m = tensor.dim %init, %c0 : tensor<?xf32>
  %a = tensor.empty(%n) : tensor<?xf16>
  %b = tensor.empty(%n) : tensor<?xf16>
  %r:3 = scf.forall (%i) in (1)
      shared_outs(%oa = %a, %om = %init, %ob = %b)
      -> (tensor<?xf16>, tensor<?xf32>, tensor<?xf16>) {
    %as = tensor.extract_slice %oa[0] [%n] [1] : tensor<?xf16> to tensor<?xf16>
    %ms = tensor.extract_slice %om[0] [%m] [1] : tensor<?xf32> to tensor<?xf32>
    %bs = tensor.extract_slice %ob[0] [%n] [1] : tensor<?xf16> to tensor<?xf16>
    %one = arith.constant 1.0 : f16
    %two = arith.constant 2.0 : f16
    %three = arith.constant 3.0 : f32
    %af = linalg.fill ins(%one : f16) outs(%as : tensor<?xf16>) -> tensor<?xf16>
    %mf = linalg.fill ins(%three : f32) outs(%ms : tensor<?xf32>) -> tensor<?xf32>
    %bf = linalg.fill ins(%two : f16) outs(%bs : tensor<?xf16>) -> tensor<?xf16>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %bf into %ob[0] [%n] [1]
          : tensor<?xf16> into tensor<?xf16>
      tensor.parallel_insert_slice %mf into %om[0] [%m] [1]
          : tensor<?xf32> into tensor<?xf32>
      tensor.parallel_insert_slice %af into %oa[0] [%n] [1]
          : tensor<?xf16> into tensor<?xf16>
    }
  }
  return %r#0, %r#1, %r#2 : tensor<?xf16>, tensor<?xf32>, tensor<?xf16>
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %loops = transform.structured.match ops{["scf.forall"]} in %root : (!transform.any_op) -> !transform.any_op
    %dispatches = transform.iree.forall_to_flow %loops : (!transform.any_op) -> !transform.any_op
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_cse to %func : !transform.any_op
    transform.yield
  }
}

// -----

// Two captured slices share a computed size. The cloned size must dominate
// both dispatch loads, and both loads must use it instead of separately
// capturing the outer size. Reversing producer discovery order is not enough
// to order a dependency shared by two branches of the captured graph.
// CHECK-LABEL: func.func @shared_capture_dependency(
// CHECK: flow.dispatch.workgroups
// CHECK: (%[[N64:.*]]: i64,
// CHECK: %[[SIZE:.*]] = arith.index_cast %[[N64]] : i64 to index
// CHECK: %[[LEFT:.*]] = iree_tensor_ext.dispatch.tensor.load {{.*}} offsets = [0, 0], sizes = [%[[SIZE]], 16], strides = [1, 1]
// CHECK: %[[RIGHT:.*]] = iree_tensor_ext.dispatch.tensor.load {{.*}} offsets = [0, 16], sizes = [%[[SIZE]], 16], strides = [1, 1]
// CHECK: tensor.extract_slice %[[LEFT]]
// CHECK: tensor.extract_slice %[[RIGHT]]
// CHECK: linalg.generic
// CHECK: arith.addf
// CHECK: iree_tensor_ext.dispatch.tensor.store
func.func @shared_capture_dependency(%input: tensor<?x32xf32>, %n64: i64) -> tensor<?x16xf32> {
  %n = arith.index_cast %n64 : i64 to index
  %left = tensor.extract_slice %input[0, 0] [%n, 16] [1, 1] : tensor<?x32xf32> to tensor<?x16xf32>
  %right = tensor.extract_slice %input[0, 16] [%n, 16] [1, 1] : tensor<?x32xf32> to tensor<?x16xf32>
  %init = tensor.empty(%n) : tensor<?x16xf32>
  %result = scf.forall (%i) in (%n) shared_outs(%out = %init) -> tensor<?x16xf32> {
    %l = tensor.extract_slice %left[%i, 0] [1, 16] [1, 1] : tensor<?x16xf32> to tensor<16xf32>
    %r = tensor.extract_slice %right[%i, 0] [1, 16] [1, 1] : tensor<?x16xf32> to tensor<16xf32>
    %o = tensor.extract_slice %out[%i, 0] [1, 16] [1, 1] : tensor<?x16xf32> to tensor<16xf32>
    %sum = linalg.generic {
        indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
        iterator_types = ["parallel"]}
        ins(%l, %r : tensor<16xf32>, tensor<16xf32>) outs(%o : tensor<16xf32>) {
    ^bb0(%lhs: f32, %rhs: f32, %unused: f32):
      %value = arith.addf %lhs, %rhs : f32
      linalg.yield %value : f32
    } -> tensor<16xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %sum into %out[%i, 0] [1, 16] [1, 1] : tensor<16xf32> into tensor<?x16xf32>
    }
  }
  return %result : tensor<?x16xf32>
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %loop = transform.structured.match ops{["scf.forall"]} in %root : (!transform.any_op) -> !transform.any_op
    %dispatch = transform.iree.forall_to_flow %loop <{clone_tensor_compute = false}> : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
