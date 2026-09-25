// RUN: iree-opt --split-input-file --iree-dispatch-creation-assign-data-tiling-encodings %s | FileCheck %s

#target = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {target_triple = "aarch64-unknown-unknown-eabi-elf"}>
module attributes {stream.affinity.default = #hal.device.affinity<@device>} {
  util.global private @device = #hal.device.target<"local", [#target]> : !hal.device
  // CHECK-LABEL: util.func public @assign_layout(
  // CHECK: %[[LHS:.*]] = iree_encoding.set_encoding %arg0
  // CHECK: %[[RHS:.*]] = iree_encoding.set_encoding %arg1
  // CHECK: %[[INIT:.*]] = iree_encoding.set_encoding %arg2
  // CHECK: %[[RESULT:.*]] = linalg.matmul ins(%[[LHS]], %[[RHS]]
  // CHECK-SAME: outs(%[[INIT]]
  // CHECK: iree_encoding.unset_encoding %[[RESULT]]
  util.func public @assign_layout(%lhs: tensor<64x128xf32>, %rhs: tensor<128x64xf32>, %init: tensor<64x64xf32>) -> tensor<64x64xf32> {
    %result = linalg.matmul ins(%lhs, %rhs : tensor<64x128xf32>, tensor<128x64xf32>) outs(%init : tensor<64x64xf32>) -> tensor<64x64xf32>
    util.return %result : tensor<64x64xf32>
  }
}

// -----

#target = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {target_triple = "aarch64-unknown-unknown-eabi-elf"}>
module attributes {iree_encoding.materialized_layout_target = #target, stream.affinity.default = #hal.device.affinity<@device>} {
  util.global private @device = #hal.device.target<"local", [#target]> : !hal.device
  // CHECK-LABEL: util.func public @preserve_layout(
  // CHECK-NOT: iree_encoding.set_encoding
  // CHECK: %[[RESULT:.*]] = linalg.matmul ins(%arg0, %arg1
  // CHECK-NOT: iree_encoding.unset_encoding
  // CHECK: util.return %[[RESULT]]
  util.func public @preserve_layout(%lhs: tensor<64x128xf32>, %rhs: tensor<128x64xf32>, %init: tensor<64x64xf32>) -> tensor<64x64xf32> {
    %result = linalg.matmul ins(%lhs, %rhs : tensor<64x128xf32>, tensor<128x64xf32>) outs(%init : tensor<64x64xf32>) -> tensor<64x64xf32>
    util.return %result : tensor<64x64xf32>
  }
}
