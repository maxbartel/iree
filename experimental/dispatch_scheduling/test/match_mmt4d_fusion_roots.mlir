// RUN: iree-opt --iree-plugin=dispatch_scheduling %s --transform-interpreter --iree-transform-dialect-drop-schedule | FileCheck %s

// Selection is read-only. The test action marks only roots which can be
// scheduled together with their initializer and pointwise epilogue.
#target = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {cpu = "generic", cpu_features = "+reserve-x18", data_layout = "e-m:e-i64:64-i128:128-n32:64-S128", ukernels = "mmt4d", target_triple = "aarch64-unknown-unknown-eabi-elf"}>
#identity = affine_map<(a, b, c, d) -> (a, b, c, d)>
#broadcast = affine_map<(a, b, c, d) -> (b, d)>
module attributes {hal.executable.target = #target} {
  // CHECK-LABEL: func.func @zero_fill(
  // CHECK: linalg.mmt4d {test.selected}
  func.func @zero_fill(%lhs: tensor<2x2x4x1xf32>, %rhs: tensor<2x2x8x1xf32>) -> tensor<2x2x4x8xf32> {
    %empty = tensor.empty() : tensor<2x2x4x8xf32>
    %zero = arith.constant 0.0 : f32
    %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<2x2x4x8xf32>) -> tensor<2x2x4x8xf32>
    %mm = linalg.mmt4d ins(%lhs, %rhs : tensor<2x2x4x1xf32>, tensor<2x2x8x1xf32>) outs(%init : tensor<2x2x4x8xf32>) -> tensor<2x2x4x8xf32>
    %destination = tensor.empty() : tensor<2x2x4x8xf32>
    %result = linalg.generic {indexing_maps = [#identity, #identity], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%mm : tensor<2x2x4x8xf32>) outs(%destination : tensor<2x2x4x8xf32>) {
    ^bb0(%x: f32, %old: f32):
      %value = arith.addf %x, %x : f32
      linalg.yield %value : f32
    } -> tensor<2x2x4x8xf32>
    return %result : tensor<2x2x4x8xf32>
  }
  // CHECK-LABEL: func.func @broadcast(
  // CHECK: linalg.mmt4d {test.selected}
  func.func @broadcast(%lhs: tensor<2x2x4x1xf32>, %rhs: tensor<2x2x8x1xf32>, %bias: tensor<2x8xf32>) -> tensor<2x2x4x8xf32> {
    %empty = tensor.empty() : tensor<2x2x4x8xf32>
    %init = linalg.generic {indexing_maps = [#broadcast, #identity], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bias : tensor<2x8xf32>) outs(%empty : tensor<2x2x4x8xf32>) {
    ^bb0(%x: f32, %unused: f32):
      linalg.yield %x : f32
    } -> tensor<2x2x4x8xf32>
    %mm = linalg.mmt4d ins(%lhs, %rhs : tensor<2x2x4x1xf32>, tensor<2x2x8x1xf32>) outs(%init : tensor<2x2x4x8xf32>) -> tensor<2x2x4x8xf32>
    %destination = tensor.empty() : tensor<2x2x4x8xf32>
    %result = linalg.generic {indexing_maps = [#identity, #identity], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%mm : tensor<2x2x4x8xf32>) outs(%destination : tensor<2x2x4x8xf32>) {
    ^bb0(%x: f32, %old: f32):
      %value = arith.addf %x, %x : f32
      linalg.yield %value : f32
    } -> tensor<2x2x4x8xf32>
    return %result : tensor<2x2x4x8xf32>
  }
  // CHECK-LABEL: func.func @dynamic_proven(
  // CHECK: linalg.mmt4d {test.selected}
  func.func @dynamic_proven(%lhs: tensor<?x2x4x1xf32>, %rhs: tensor<2x2x8x1xf32>) -> tensor<?x2x4x8xf32> {
    %c0 = arith.constant 0 : index
    %m = tensor.dim %lhs, %c0 : tensor<?x2x4x1xf32>
    %empty = tensor.empty(%m) : tensor<?x2x4x8xf32>
    %zero = arith.constant 0.0 : f32
    %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<?x2x4x8xf32>) -> tensor<?x2x4x8xf32>
    %mm = linalg.mmt4d ins(%lhs, %rhs : tensor<?x2x4x1xf32>, tensor<2x2x8x1xf32>) outs(%init : tensor<?x2x4x8xf32>) -> tensor<?x2x4x8xf32>
    %destination = tensor.empty(%m) : tensor<?x2x4x8xf32>
    %result = linalg.generic {indexing_maps = [#identity, #identity], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%mm : tensor<?x2x4x8xf32>) outs(%destination : tensor<?x2x4x8xf32>) {
    ^bb0(%x: f32, %old: f32):
      %value = arith.addf %x, %x : f32
      linalg.yield %value : f32
    } -> tensor<?x2x4x8xf32>
    return %result : tensor<?x2x4x8xf32>
  }
  // CHECK-LABEL: func.func @contraction_input_one(
  // CHECK: linalg.mmt4d {test.selected}
  func.func @contraction_input_one(%lhs: tensor<2x2x4x1xf32>, %rhs: tensor<2x2x8x1xf32>) -> tensor<2x2x4x8xf32> {
    %empty = tensor.empty() : tensor<2x2x4x8xf32>
    %zero = arith.constant 0.0 : f32
    %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<2x2x4x8xf32>) -> tensor<2x2x4x8xf32>
    %mm = linalg.mmt4d ins(%lhs, %rhs : tensor<2x2x4x1xf32>, tensor<2x2x8x1xf32>) outs(%init : tensor<2x2x4x8xf32>) -> tensor<2x2x4x8xf32>
    %destination = tensor.empty() : tensor<2x2x4x8xf32>
    %result = linalg.generic {indexing_maps = [#identity, #identity, #identity], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%init, %mm : tensor<2x2x4x8xf32>, tensor<2x2x4x8xf32>) outs(%destination : tensor<2x2x4x8xf32>) {
    ^bb0(%ignored: f32, %x: f32, %old: f32):
      %value = arith.addf %x, %x : f32
      linalg.yield %value : f32
    } -> tensor<2x2x4x8xf32>
    return %result : tensor<2x2x4x8xf32>
  }
  // CHECK-LABEL: func.func @nonzero_fill(
  // CHECK-NOT: test.selected
  // CHECK: linalg.mmt4d ins(
  // CHECK-NOT: test.selected
  // CHECK: return
  func.func @nonzero_fill(%lhs: tensor<2x2x4x1xf32>, %rhs: tensor<2x2x8x1xf32>) -> tensor<2x2x4x8xf32> {
    %empty = tensor.empty() : tensor<2x2x4x8xf32>
    %zero = arith.constant 1.0 : f32
    %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<2x2x4x8xf32>) -> tensor<2x2x4x8xf32>
    %mm = linalg.mmt4d ins(%lhs, %rhs : tensor<2x2x4x1xf32>, tensor<2x2x8x1xf32>) outs(%init : tensor<2x2x4x8xf32>) -> tensor<2x2x4x8xf32>
    %destination = tensor.empty() : tensor<2x2x4x8xf32>
    %result = linalg.generic {indexing_maps = [#identity, #identity], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%mm : tensor<2x2x4x8xf32>) outs(%destination : tensor<2x2x4x8xf32>) {
    ^bb0(%x: f32, %old: f32):
      %value = arith.addf %x, %x : f32
      linalg.yield %value : f32
    } -> tensor<2x2x4x8xf32>
    return %result : tensor<2x2x4x8xf32>
  }
  // CHECK-LABEL: func.func @needs_math_lowering(
  // CHECK-NOT: test.selected
  // CHECK: linalg.mmt4d ins(
  // CHECK-NOT: test.selected
  // CHECK: return
  func.func @needs_math_lowering(%lhs: tensor<2x2x4x1xf32>, %rhs: tensor<2x2x8x1xf32>) -> tensor<2x2x4x8xf32> {
    %empty = tensor.empty() : tensor<2x2x4x8xf32>
    %zero = arith.constant 0.0 : f32
    %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<2x2x4x8xf32>) -> tensor<2x2x4x8xf32>
    %mm = linalg.mmt4d ins(%lhs, %rhs : tensor<2x2x4x1xf32>, tensor<2x2x8x1xf32>) outs(%init : tensor<2x2x4x8xf32>) -> tensor<2x2x4x8xf32>
    %destination = tensor.empty() : tensor<2x2x4x8xf32>
    %result = linalg.generic {indexing_maps = [#identity, #identity], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%mm : tensor<2x2x4x8xf32>) outs(%destination : tensor<2x2x4x8xf32>) {
    ^bb0(%x: f32, %old: f32):
      %value = math.exp %x : f32
      linalg.yield %value : f32
    } -> tensor<2x2x4x8xf32>
    return %result : tensor<2x2x4x8xf32>
  }
  // CHECK-LABEL: func.func @multiple_uses(
  // CHECK-NOT: test.selected
  // CHECK: linalg.mmt4d ins(
  // CHECK-NOT: test.selected
  // CHECK: return
  func.func @multiple_uses(%lhs: tensor<2x2x4x1xf32>, %rhs: tensor<2x2x8x1xf32>) -> (tensor<2x2x4x8xf32>, tensor<2x2x4x8xf32>) {
    %empty = tensor.empty() : tensor<2x2x4x8xf32>
    %zero = arith.constant 0.0 : f32
    %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<2x2x4x8xf32>) -> tensor<2x2x4x8xf32>
    %mm = linalg.mmt4d ins(%lhs, %rhs : tensor<2x2x4x1xf32>, tensor<2x2x8x1xf32>) outs(%init : tensor<2x2x4x8xf32>) -> tensor<2x2x4x8xf32>
    %destination = tensor.empty() : tensor<2x2x4x8xf32>
    %result = linalg.generic {indexing_maps = [#identity, #identity], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%mm : tensor<2x2x4x8xf32>) outs(%destination : tensor<2x2x4x8xf32>) {
    ^bb0(%x: f32, %old: f32):
      %value = arith.addf %x, %x : f32
      linalg.yield %value : f32
    } -> tensor<2x2x4x8xf32>
    return %result, %mm : tensor<2x2x4x8xf32>, tensor<2x2x4x8xf32>
  }
  // CHECK-LABEL: func.func @reads_old_output(
  // CHECK-NOT: test.selected
  // CHECK: linalg.mmt4d ins(
  // CHECK-NOT: test.selected
  // CHECK: return
  func.func @reads_old_output(%lhs: tensor<2x2x4x1xf32>, %rhs: tensor<2x2x8x1xf32>) -> tensor<2x2x4x8xf32> {
    %empty = tensor.empty() : tensor<2x2x4x8xf32>
    %zero = arith.constant 0.0 : f32
    %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<2x2x4x8xf32>) -> tensor<2x2x4x8xf32>
    %mm = linalg.mmt4d ins(%lhs, %rhs : tensor<2x2x4x1xf32>, tensor<2x2x8x1xf32>) outs(%init : tensor<2x2x4x8xf32>) -> tensor<2x2x4x8xf32>
    %destination = tensor.empty() : tensor<2x2x4x8xf32>
    %result = linalg.generic {indexing_maps = [#identity, #identity], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%mm : tensor<2x2x4x8xf32>) outs(%destination : tensor<2x2x4x8xf32>) {
    ^bb0(%x: f32, %old: f32):
      %value = arith.addf %x, %old : f32
      linalg.yield %value : f32
    } -> tensor<2x2x4x8xf32>
    return %result : tensor<2x2x4x8xf32>
  }
  // CHECK-LABEL: func.func @unsupported_element_type(
  // CHECK-NOT: test.selected
  // CHECK: linalg.mmt4d ins(
  // CHECK-NOT: test.selected
  // CHECK: return
  func.func @unsupported_element_type(%lhs: tensor<2x2x4x1xf16>, %rhs: tensor<2x2x8x1xf16>) -> tensor<2x2x4x8xf16> {
    %empty = tensor.empty() : tensor<2x2x4x8xf16>
    %zero = arith.constant 0.0 : f16
    %init = linalg.fill ins(%zero : f16) outs(%empty : tensor<2x2x4x8xf16>) -> tensor<2x2x4x8xf16>
    %mm = linalg.mmt4d ins(%lhs, %rhs : tensor<2x2x4x1xf16>, tensor<2x2x8x1xf16>) outs(%init : tensor<2x2x4x8xf16>) -> tensor<2x2x4x8xf16>
    %destination = tensor.empty() : tensor<2x2x4x8xf16>
    %result = linalg.generic {indexing_maps = [#identity, #identity], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%mm : tensor<2x2x4x8xf16>) outs(%destination : tensor<2x2x4x8xf16>) {
    ^bb0(%x: f16, %old: f16):
      %value = arith.addf %x, %x : f16
      linalg.yield %value : f16
    } -> tensor<2x2x4x8xf16>
    return %result : tensor<2x2x4x8xf16>
  }
  // CHECK-LABEL: func.func @dynamic_unproven(
  // CHECK-NOT: test.selected
  // CHECK: linalg.mmt4d ins(
  // CHECK-NOT: test.selected
  // CHECK: return
  func.func @dynamic_unproven(%lhs: tensor<?x2x4x1xf32>, %rhs: tensor<2x2x8x1xf32>, %other: index) -> tensor<?x2x4x8xf32> {
    %c0 = arith.constant 0 : index
    %m = tensor.dim %lhs, %c0 : tensor<?x2x4x1xf32>
    %empty = tensor.empty(%m) : tensor<?x2x4x8xf32>
    %zero = arith.constant 0.0 : f32
    %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<?x2x4x8xf32>) -> tensor<?x2x4x8xf32>
    %mm = linalg.mmt4d ins(%lhs, %rhs : tensor<?x2x4x1xf32>, tensor<2x2x8x1xf32>) outs(%init : tensor<?x2x4x8xf32>) -> tensor<?x2x4x8xf32>
    %destination = tensor.empty(%other) : tensor<?x2x4x8xf32>
    %result = linalg.generic {indexing_maps = [#identity, #identity], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%mm : tensor<?x2x4x8xf32>) outs(%destination : tensor<?x2x4x8xf32>) {
    ^bb0(%x: f32, %old: f32):
      %value = arith.addf %x, %x : f32
      linalg.yield %value : f32
    } -> tensor<?x2x4x8xf32>
    return %result : tensor<?x2x4x8xf32>
  }
  module @owned attributes {iree_payload.compute} {
  // CHECK-LABEL: func.func @owned_function(
  // CHECK-NOT: test.selected
  // CHECK: linalg.mmt4d ins(
  // CHECK-NOT: test.selected
  // CHECK: return
  func.func @owned_function(%lhs: tensor<2x2x4x1xf32>, %rhs: tensor<2x2x8x1xf32>) -> tensor<2x2x4x8xf32> {
    %empty = tensor.empty() : tensor<2x2x4x8xf32>
    %zero = arith.constant 0.0 : f32
    %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<2x2x4x8xf32>) -> tensor<2x2x4x8xf32>
    %mm = linalg.mmt4d ins(%lhs, %rhs : tensor<2x2x4x1xf32>, tensor<2x2x8x1xf32>) outs(%init : tensor<2x2x4x8xf32>) -> tensor<2x2x4x8xf32>
    %destination = tensor.empty() : tensor<2x2x4x8xf32>
    %result = linalg.generic {indexing_maps = [#identity, #identity], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%mm : tensor<2x2x4x8xf32>) outs(%destination : tensor<2x2x4x8xf32>) {
    ^bb0(%x: f32, %old: f32):
      %value = arith.addf %x, %x : f32
      linalg.yield %value : f32
    } -> tensor<2x2x4x8xf32>
    return %result : tensor<2x2x4x8xf32>
  }
  }
  module @disabled attributes {hal.executable.target = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {cpu = "generic", cpu_features = "+reserve-x18", data_layout = "e-m:e-i64:64-i128:128-n32:64-S128", ukernels = "none", target_triple = "aarch64-unknown-unknown-eabi-elf"}>} {
    // CHECK-LABEL: func.func @disabled_microkernel(
    // CHECK-NOT: test.selected
    // CHECK: linalg.mmt4d ins(
    // CHECK-NOT: test.selected
    // CHECK: return
  func.func @disabled_microkernel(%lhs: tensor<2x2x4x1xf32>, %rhs: tensor<2x2x8x1xf32>) -> tensor<2x2x4x8xf32> {
    %empty = tensor.empty() : tensor<2x2x4x8xf32>
    %zero = arith.constant 0.0 : f32
    %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<2x2x4x8xf32>) -> tensor<2x2x4x8xf32>
    %mm = linalg.mmt4d ins(%lhs, %rhs : tensor<2x2x4x1xf32>, tensor<2x2x8x1xf32>) outs(%init : tensor<2x2x4x8xf32>) -> tensor<2x2x4x8xf32>
    %destination = tensor.empty() : tensor<2x2x4x8xf32>
    %result = linalg.generic {indexing_maps = [#identity, #identity], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%mm : tensor<2x2x4x8xf32>) outs(%destination : tensor<2x2x4x8xf32>) {
    ^bb0(%x: f32, %old: f32):
      %value = arith.addf %x, %x : f32
      linalg.yield %value : f32
    } -> tensor<2x2x4x8xf32>
    return %result : tensor<2x2x4x8xf32>
  }

  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @match0(%function: !transform.any_op {transform.readonly}) -> !transform.any_op {
      %roots = transform.iree.cpu.match_mmt4d_fusion_roots %function <{contraction_input = 0}> : (!transform.any_op) -> !transform.any_op
      transform.yield %roots : !transform.any_op
    }
    transform.named_sequence @match1(%function: !transform.any_op {transform.readonly}) -> !transform.any_op {
      %roots = transform.iree.cpu.match_mmt4d_fusion_roots %function <{contraction_input = 1}> : (!transform.any_op) -> !transform.any_op
      transform.yield %roots : !transform.any_op
    }
    transform.named_sequence @mark(%roots: !transform.any_op {transform.readonly}) {
      transform.annotate %roots "test.selected" : !transform.any_op
      transform.yield
    }
    transform.named_sequence @__transform_main(%root: !transform.any_op {transform.consumed}) {
      %selected = transform.foreach_match in %root @match0 -> @mark, @match1 -> @mark : (!transform.any_op) -> !transform.any_op
      transform.yield
    }
  }
}
