// RUN: iree-opt --iree-plugin=dispatch_scheduling %s --transform-interpreter | FileCheck %s --check-prefix=DEFAULT --implicit-check-not=lowering_config --implicit-check-not=translation_info
// RUN: iree-opt --iree-plugin=dispatch_scheduling %s --transform-interpreter --iree-llvmcpu-number-of-threads=1 | FileCheck %s --check-prefix=ONE
// RUN: iree-opt --iree-plugin=dispatch_scheduling %s --transform-interpreter --iree-llvmcpu-matmul-tile-bytes=4096 | FileCheck %s --check-prefix=SMALL
// RUN: iree-opt --iree-plugin=dispatch_scheduling %s --transform-interpreter --iree-llvmcpu-select-lowering-strategy | FileCheck %s --check-prefix=CLASSIC

// Querying multiple roots preserves their order and does not configure them.
// Thread-count and memory-budget options affect the same heuristic used by the
// classic selector. The last run compares the queried values with its configs.
// DEFAULT-LABEL: func.func @small(
// DEFAULT: linalg.mmt4d {test.m = 1 : i64, test.n = 1 : i64}
// DEFAULT-LABEL: func.func @large(
// DEFAULT: linalg.mmt4d {test.m = 4 : i64, test.n = 8 : i64}
// DEFAULT-LABEL: func.func @dynamic(
// DEFAULT: linalg.mmt4d {test.m = 2 : i64, test.n = 4 : i64}
// ONE-LABEL: func.func @small(
// ONE: linalg.mmt4d {test.m = 2 : i64, test.n = 3 : i64}
// ONE-LABEL: func.func @large(
// ONE: linalg.mmt4d {test.m = 16 : i64, test.n = 16 : i64}
// ONE-LABEL: func.func @dynamic(
// ONE: linalg.mmt4d {test.m = 2 : i64, test.n = 4 : i64}
// SMALL-LABEL: func.func @small(
// SMALL: linalg.mmt4d {test.m = 1 : i64, test.n = 1 : i64}
// SMALL-LABEL: func.func @large(
// SMALL: linalg.mmt4d {test.m = 1 : i64, test.n = 1 : i64}
// SMALL-LABEL: func.func @dynamic(
// SMALL: linalg.mmt4d {test.m = 1 : i64, test.n = 1 : i64}
// CLASSIC-DAG: #[[$S:[^ ]+]] = #iree_cpu.lowering_config<distribution = [1, 1, 0, 0, 0, 0]
// CLASSIC-DAG: #[[$L:[^ ]+]] = #iree_cpu.lowering_config<distribution = [4, 8, 0, 0, 0, 0]
// CLASSIC-DAG: #[[$D:[^ ]+]] = #iree_cpu.lowering_config<distribution = [2, 4, 0, 0, 0, 0]
// CLASSIC-LABEL: func.func @small(
// CLASSIC: linalg.mmt4d {lowering_config = #[[$S]], test.m = 1 : i64, test.n = 1 : i64}
// CLASSIC-LABEL: func.func @large(
// CLASSIC: linalg.mmt4d {lowering_config = #[[$L]], test.m = 4 : i64, test.n = 8 : i64}
// CLASSIC-LABEL: func.func @dynamic(
// CLASSIC: linalg.mmt4d {lowering_config = #[[$D]], test.m = 2 : i64, test.n = 4 : i64}

#target = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {cpu = "generic", cpu_features = "+reserve-x18", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>, max_stack_allocation_size = 32768 : i64, native_vector_size = 16 : i64, target_triple = "aarch64-unknown-unknown-eabi-elf"}>
module attributes {hal.executable.target = #target} {
func.func @small(%a: tensor<2x4x8x1xf32>, %b: tensor<3x4x8x1xf32>, %o: tensor<2x3x8x8xf32>) -> tensor<2x3x8x8xf32> {
    %r = linalg.mmt4d ins(%a, %b : tensor<2x4x8x1xf32>, tensor<3x4x8x1xf32>) outs(%o : tensor<2x3x8x8xf32>) -> tensor<2x3x8x8xf32>
    return %r : tensor<2x3x8x8xf32>
  }
func.func @large(%a: tensor<16x128x8x1xf32>, %b: tensor<16x128x8x1xf32>, %o: tensor<16x16x8x8xf32>) -> tensor<16x16x8x8xf32> {
    %r = linalg.mmt4d ins(%a, %b : tensor<16x128x8x1xf32>, tensor<16x128x8x1xf32>) outs(%o : tensor<16x16x8x8xf32>) -> tensor<16x16x8x8xf32>
    return %r : tensor<16x16x8x8xf32>
  }
func.func @dynamic(%a: tensor<?x?x8x1xf32>, %b: tensor<?x?x8x1xf32>, %o: tensor<?x?x8x8xf32>) -> tensor<?x?x8x8xf32> {
    %r = linalg.mmt4d ins(%a, %b : tensor<?x?x8x1xf32>, tensor<?x?x8x1xf32>) outs(%o : tensor<?x?x8x8xf32>) -> tensor<?x?x8x8xf32>
    return %r : tensor<?x?x8x8xf32>
  }
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %roots = transform.structured.match ops{["linalg.mmt4d"]} in %root : (!transform.any_op) -> !transform.any_op
    %m, %n = transform.iree.cpu.get_mmt4d_workgroup_tiles %roots : (!transform.any_op) -> (!transform.param<i64>, !transform.param<i64>)
    transform.annotate %roots "test.m" = %m : !transform.any_op, !transform.param<i64>
    transform.annotate %roots "test.n" = %n : !transform.any_op, !transform.param<i64>
    %none = transform.structured.match ops{["linalg.matmul"]} in %root : (!transform.any_op) -> !transform.any_op
    %empty_m, %empty_n = transform.iree.cpu.get_mmt4d_workgroup_tiles %none : (!transform.any_op) -> (!transform.param<i64>, !transform.param<i64>)
    transform.yield
  }
}
}
