// RUN: iree-opt --iree-plugin=dispatch_scheduling %s --split-input-file --transform-interpreter --verify-diagnostics

#target = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {cpu = "generic", cpu_features = "+reserve-x18", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>, max_stack_allocation_size = 32768 : i64, native_vector_size = 16 : i64, target_triple = "aarch64-unknown-unknown-eabi-elf"}>
module {
func.func @no_target(%a: tensor<2x4x8x1xf32>, %b: tensor<3x4x8x1xf32>, %o: tensor<2x3x8x8xf32>) -> tensor<2x3x8x8xf32> {
  %r = linalg.mmt4d  ins(%a, %b : tensor<2x4x8x1xf32>, tensor<3x4x8x1xf32>) outs(%o : tensor<2x3x8x8xf32>) -> tensor<2x3x8x8xf32>
  return %r : tensor<2x3x8x8xf32>
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %roots = transform.structured.match ops{["linalg.mmt4d"]} in %root : (!transform.any_op) -> !transform.any_op
    // expected-error @+1 {{expected an unconfigured mmt4d with a resolved LLVM CPU target}}
    %m, %n = transform.iree.cpu.get_mmt4d_workgroup_tiles %roots : (!transform.any_op) -> (!transform.param<i64>, !transform.param<i64>)
    transform.yield
  }
}
}

// -----
#target = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {cpu = "generic", cpu_features = "+reserve-x18", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>, max_stack_allocation_size = 32768 : i64, native_vector_size = 16 : i64, target_triple = "aarch64-unknown-unknown-eabi-elf"}>
module attributes {hal.executable.target = #target} {
func.func @buffer_form(%a: memref<2x4x8x1xf32>, %b: memref<3x4x8x1xf32>, %o: memref<2x3x8x8xf32>) {
  linalg.mmt4d  ins(%a, %b : memref<2x4x8x1xf32>, memref<3x4x8x1xf32>) outs(%o : memref<2x3x8x8xf32>)
  return
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %roots = transform.structured.match ops{["linalg.mmt4d"]} in %root : (!transform.any_op) -> !transform.any_op
    // expected-error @+1 {{expected an unconfigured mmt4d with a resolved LLVM CPU target}}
    %m, %n = transform.iree.cpu.get_mmt4d_workgroup_tiles %roots : (!transform.any_op) -> (!transform.param<i64>, !transform.param<i64>)
    transform.yield
  }
}
}

// -----
#target = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {cpu = "generic", cpu_features = "+reserve-x18", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>, max_stack_allocation_size = 32768 : i64, native_vector_size = 16 : i64, target_triple = "aarch64-unknown-unknown-eabi-elf"}>
module attributes {hal.executable.target = #target} {
func.func @dynamic_inner(%a: tensor<2x4x?x1xf32>, %b: tensor<3x4x8x1xf32>, %o: tensor<2x3x?x8xf32>) -> tensor<2x3x?x8xf32> {
  %r = linalg.mmt4d  ins(%a, %b : tensor<2x4x?x1xf32>, tensor<3x4x8x1xf32>) outs(%o : tensor<2x3x?x8xf32>) -> tensor<2x3x?x8xf32>
  return %r : tensor<2x3x?x8xf32>
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %roots = transform.structured.match ops{["linalg.mmt4d"]} in %root : (!transform.any_op) -> !transform.any_op
    // expected-error @+1 {{expected nonempty outer dimensions and static positive inner tiles}}
    %m, %n = transform.iree.cpu.get_mmt4d_workgroup_tiles %roots : (!transform.any_op) -> (!transform.param<i64>, !transform.param<i64>)
    transform.yield
  }
}
}

// -----
#target = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {cpu = "generic", cpu_features = "+reserve-x18", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>, max_stack_allocation_size = 32768 : i64, native_vector_size = 16 : i64, target_triple = "aarch64-unknown-unknown-eabi-elf"}>
module attributes {hal.executable.target = #target} {
func.func @empty_outer(%a: tensor<0x4x8x1xf32>, %b: tensor<3x4x8x1xf32>, %o: tensor<0x3x8x8xf32>) -> tensor<0x3x8x8xf32> {
  %r = linalg.mmt4d  ins(%a, %b : tensor<0x4x8x1xf32>, tensor<3x4x8x1xf32>) outs(%o : tensor<0x3x8x8xf32>) -> tensor<0x3x8x8xf32>
  return %r : tensor<0x3x8x8xf32>
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %roots = transform.structured.match ops{["linalg.mmt4d"]} in %root : (!transform.any_op) -> !transform.any_op
    // expected-error @+1 {{expected nonempty outer dimensions and static positive inner tiles}}
    %m, %n = transform.iree.cpu.get_mmt4d_workgroup_tiles %roots : (!transform.any_op) -> (!transform.param<i64>, !transform.param<i64>)
    transform.yield
  }
}
}

// -----
#target = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {cpu = "generic", cpu_features = "+reserve-x18", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>, max_stack_allocation_size = 32768 : i64, native_vector_size = 16 : i64, target_triple = "aarch64-unknown-unknown-eabi-elf"}>
module attributes {hal.executable.target = #target} {
func.func @already_configured(%a: tensor<2x4x8x1xf32>, %b: tensor<3x4x8x1xf32>, %o: tensor<2x3x8x8xf32>) -> tensor<2x3x8x8xf32> {
  %r = linalg.mmt4d {lowering_config = #iree_cpu.lowering_config<distribution = [1, 1, 0, 0, 0, 0]>} ins(%a, %b : tensor<2x4x8x1xf32>, tensor<3x4x8x1xf32>) outs(%o : tensor<2x3x8x8xf32>) -> tensor<2x3x8x8xf32>
  return %r : tensor<2x3x8x8xf32>
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %roots = transform.structured.match ops{["linalg.mmt4d"]} in %root : (!transform.any_op) -> !transform.any_op
    // expected-error @+1 {{expected an unconfigured mmt4d with a resolved LLVM CPU target}}
    %m, %n = transform.iree.cpu.get_mmt4d_workgroup_tiles %roots : (!transform.any_op) -> (!transform.param<i64>, !transform.param<i64>)
    transform.yield
  }
}
}
