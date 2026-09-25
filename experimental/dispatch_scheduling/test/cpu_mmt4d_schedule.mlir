// RUN: iree-opt --iree-plugin=dispatch_scheduling %s --iree-dispatch-scheduling-select -o %t
// RUN: FileCheck %s --input-file=%t --implicit-check-not="llvm.func @entry" --implicit-check-not=flow.dispatch.workgroups --implicit-check-not=memref.copy --implicit-check-not=llvm.alloca
// RUN: iree-opt --iree-plugin=dispatch_scheduling %t --iree-dispatch-scheduling-select -o %t.repeated
// RUN: diff %t %t.repeated
// RUN: iree-opt --iree-plugin=dispatch_scheduling %s --iree-dispatch-scheduling-select=stop-after-selection -o %t.selected
// RUN: FileCheck %s --input-file=%t.selected --check-prefix=SELECTED --implicit-check-not="llvm.func @entry" --implicit-check-not=iree_payload.schedule_completed --implicit-check-not=iree_payload.preserved
// RUN: iree-opt --iree-plugin=dispatch_scheduling %t.selected --iree-dispatch-scheduling-execute -o %t.resumed
// RUN: diff %t %t.resumed
// RUN: iree-opt --iree-plugin=dispatch_scheduling %t.selected --iree-dispatch-scheduling-execute --mlir-disable-threading -o %t.serial
// RUN: diff %t.resumed %t.serial
// RUN: not iree-opt --iree-plugin=dispatch_scheduling %s --iree-dispatch-scheduling-select=library-file=%S/Inputs/unlowered_schedule.mlir 2>&1 | FileCheck %s --check-prefix=UNLOWERED
// RUN: not iree-opt --iree-plugin=dispatch_scheduling %s --iree-dispatch-scheduling-select=library-file=%S/Inputs/failing_schedule.mlir 2>&1 | FileCheck %s --check-prefix=FAILURE
// RUN: not iree-opt --iree-plugin=dispatch_scheduling %s --iree-dispatch-scheduling-select=library-file=%t.missing 2>&1 | FileCheck %s --check-prefix=MISSING

// UNLOWERED: dispatch schedule must produce one owned LLVM compute module
// FAILURE: selected dispatch scheduling schedule failed
// MISSING: cannot load dispatch transform library

// SELECTED-LABEL: flow.executable private @fused_dispatch_0
// SELECTED: iree_payload.schedule = {configuration = {}, entry_point = "cpu_mmt4d"}
// SELECTED-SAME: iree_payload.schedule_environment = {library_sha256 = "{{([0-9a-f]{64})}}", target =
// SELECTED: iree_payload.region
// SELECTED: linalg.mmt4d
// SELECTED-LABEL: flow.executable private @different_dispatch_0
// SELECTED: iree_payload.region
// SELECTED: linalg.mmt4d

// Three disjoint groups across two entry points share one owned LLVM body.
// The dispatch invocations and their distinct tensor operands remain intact.
// Unmatched operations remain available to classic codegen.

// CHECK-LABEL: flow.executable private @fused_dispatch_0
// CHECK: iree_payload.region
// CHECK: builtin.module @compute attributes {
// CHECK-SAME: iree_payload.lowered
// CHECK-SAME: iree_payload.schedule_sha256
// CHECK: llvm.func @entry
// CHECK: llvm.call @iree_uk_mmt4d
// CHECK: llvm.fadd
// CHECK: iree_payload.call
// CHECK-LABEL: func.func @fused(
// CHECK-NOT: linalg.mmt4d
// CHECK: flow.dispatch @fused_dispatch_0::@fused_dispatch_0
// CHECK: return
// CHECK-LABEL: func.func @multiple(
// CHECK: flow.dispatch @fused_dispatch_0::@fused_dispatch_0
// CHECK: flow.dispatch @fused_dispatch_0::@fused_dispatch_0
// CHECK-NOT: linalg.mmt4d
// CHECK: return
// A different epilogue must keep its own numerical body.
// CHECK-LABEL: flow.executable private @different_dispatch_0
// CHECK: llvm.func @entry
// CHECK: llvm.call @iree_uk_mmt4d
// CHECK: llvm.fsub
// CHECK-LABEL: func.func @different(
// CHECK: flow.dispatch @different_dispatch_0::@different_dispatch_0
// CHECK: return
// CHECK-LABEL: func.func private @external()
// CHECK-LABEL: func.func @empty()
// CHECK-NEXT: return
// CHECK-LABEL: func.func @unsupported_f16(
// CHECK: linalg.mmt4d
// CHECK: linalg.generic
// CHECK-NOT: flow.dispatch.workgroups
// CHECK: return

#target = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {cpu = "generic", ukernels = "all", cpu_features = "+reserve-x18", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>, max_stack_allocation_size = 32768 : i64, native_vector_size = 16 : i64, target_triple = "aarch64-unknown-unknown-eabi-elf"}>
module attributes {hal.executable.target = #target} {
func.func @fused(%a: tensor<16x128x8x1xf32>, %b: tensor<16x128x8x1xf32>, %bias: tensor<16x16x8x8xf32>, %out: tensor<16x16x8x8xf32>) -> tensor<16x16x8x8xf32> {
  %zero = arith.constant 0.0 : f32
  %init = linalg.fill ins(%zero : f32) outs(%out : tensor<16x16x8x8xf32>) -> tensor<16x16x8x8xf32>
  %mm = linalg.mmt4d ins(%a, %b : tensor<16x128x8x1xf32>, tensor<16x128x8x1xf32>) outs(%init : tensor<16x16x8x8xf32>) -> tensor<16x16x8x8xf32>
  %empty = tensor.empty() : tensor<16x16x8x8xf32>
  %sum = linalg.generic {
    indexing_maps = [affine_map<(m,n,i,j)->(m,n,i,j)>, affine_map<(m,n,i,j)->(m,n,i,j)>, affine_map<(m,n,i,j)->(m,n,i,j)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    ins(%bias, %mm : tensor<16x16x8x8xf32>, tensor<16x16x8x8xf32>) outs(%empty : tensor<16x16x8x8xf32>) {
  ^bb0(%bias_value: f32, %acc: f32, %unused: f32):
    %v = arith.addf %acc, %bias_value : f32
    linalg.yield %v : f32
  } -> tensor<16x16x8x8xf32>
  return %sum : tensor<16x16x8x8xf32>
}
func.func @multiple(%a: tensor<16x128x8x1xf32>, %b: tensor<16x128x8x1xf32>, %bias: tensor<16x16x8x8xf32>, %out: tensor<16x16x8x8xf32>) -> (tensor<16x16x8x8xf32>, tensor<16x16x8x8xf32>) {
  %zero = arith.constant 0.0 : f32
  %init = linalg.fill ins(%zero : f32) outs(%out : tensor<16x16x8x8xf32>) -> tensor<16x16x8x8xf32>
  %mm = linalg.mmt4d ins(%a, %b : tensor<16x128x8x1xf32>, tensor<16x128x8x1xf32>) outs(%init : tensor<16x16x8x8xf32>) -> tensor<16x16x8x8xf32>
  %empty = tensor.empty() : tensor<16x16x8x8xf32>
  %sum = linalg.generic {
    indexing_maps = [affine_map<(m,n,i,j)->(m,n,i,j)>, affine_map<(m,n,i,j)->(m,n,i,j)>, affine_map<(m,n,i,j)->(m,n,i,j)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    ins(%bias, %mm : tensor<16x16x8x8xf32>, tensor<16x16x8x8xf32>) outs(%empty : tensor<16x16x8x8xf32>) {
  ^bb0(%bias_value: f32, %acc: f32, %unused: f32):
    %v = arith.addf %acc, %bias_value : f32
    linalg.yield %v : f32
  } -> tensor<16x16x8x8xf32>
  %mm2 = linalg.mmt4d ins(%b, %a : tensor<16x128x8x1xf32>, tensor<16x128x8x1xf32>) outs(%init : tensor<16x16x8x8xf32>) -> tensor<16x16x8x8xf32>
  %empty2 = tensor.empty() : tensor<16x16x8x8xf32>
  %sum2 = linalg.generic {
    indexing_maps = [affine_map<(m,n,i,j)->(m,n,i,j)>, affine_map<(m,n,i,j)->(m,n,i,j)>, affine_map<(m,n,i,j)->(m,n,i,j)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    ins(%bias, %mm2 : tensor<16x16x8x8xf32>, tensor<16x16x8x8xf32>) outs(%empty2 : tensor<16x16x8x8xf32>) {
  ^bb0(%bias_value: f32, %acc: f32, %unused: f32):
    %v = arith.addf %acc, %bias_value : f32
    linalg.yield %v : f32
  } -> tensor<16x16x8x8xf32>
  return %sum, %sum2 : tensor<16x16x8x8xf32>, tensor<16x16x8x8xf32>
}
func.func @different(%a: tensor<16x128x8x1xf32>, %b: tensor<16x128x8x1xf32>, %bias: tensor<16x16x8x8xf32>, %out: tensor<16x16x8x8xf32>) -> tensor<16x16x8x8xf32> {
  %zero = arith.constant 0.0 : f32
  %init = linalg.fill ins(%zero : f32) outs(%out : tensor<16x16x8x8xf32>) -> tensor<16x16x8x8xf32>
  %mm = linalg.mmt4d ins(%a, %b : tensor<16x128x8x1xf32>, tensor<16x128x8x1xf32>) outs(%init : tensor<16x16x8x8xf32>) -> tensor<16x16x8x8xf32>
  %empty = tensor.empty() : tensor<16x16x8x8xf32>
  %sum = linalg.generic {
    indexing_maps = [affine_map<(m,n,i,j)->(m,n,i,j)>, affine_map<(m,n,i,j)->(m,n,i,j)>, affine_map<(m,n,i,j)->(m,n,i,j)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    ins(%bias, %mm : tensor<16x16x8x8xf32>, tensor<16x16x8x8xf32>) outs(%empty : tensor<16x16x8x8xf32>) {
  ^bb0(%bias_value: f32, %acc: f32, %unused: f32):
    %v = arith.subf %acc, %bias_value : f32
    linalg.yield %v : f32
  } -> tensor<16x16x8x8xf32>
  return %sum : tensor<16x16x8x8xf32>
}
func.func private @external()
func.func @empty() { return }
func.func @unsupported_f16(%a: tensor<16x128x8x1xf16>, %b: tensor<16x128x8x1xf16>, %bias: tensor<16x16x8x8xf16>, %out: tensor<16x16x8x8xf16>) -> tensor<16x16x8x8xf16> {
  %zero = arith.constant 0.0 : f16
  %init = linalg.fill ins(%zero : f16) outs(%out : tensor<16x16x8x8xf16>) -> tensor<16x16x8x8xf16>
  %mm = linalg.mmt4d ins(%a, %b : tensor<16x128x8x1xf16>, tensor<16x128x8x1xf16>) outs(%init : tensor<16x16x8x8xf16>) -> tensor<16x16x8x8xf16>
  %empty = tensor.empty() : tensor<16x16x8x8xf16>
  %sum = linalg.generic {
    indexing_maps = [affine_map<(m,n,i,j)->(m,n,i,j)>, affine_map<(m,n,i,j)->(m,n,i,j)>, affine_map<(m,n,i,j)->(m,n,i,j)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    ins(%bias, %mm : tensor<16x16x8x8xf16>, tensor<16x16x8x8xf16>) outs(%empty : tensor<16x16x8x8xf16>) {
  ^bb0(%bias_value: f16, %acc: f16, %unused: f16):
    %v = arith.addf %acc, %bias_value : f16
    linalg.yield %v : f16
  } -> tensor<16x16x8x8xf16>
  return %sum : tensor<16x16x8x8xf16>
}

// CHECK-LABEL: func.func @unsupported_erf(
// CHECK: linalg.mmt4d
// CHECK: math.erf
// CHECK-NOT: flow.dispatch.workgroups
// CHECK: return

func.func @unsupported_erf(%a: tensor<16x128x8x1xf32>, %b: tensor<16x128x8x1xf32>, %bias: tensor<16x16x8x8xf32>, %out: tensor<16x16x8x8xf32>) -> tensor<16x16x8x8xf32> {
  %zero = arith.constant 0.0 : f32
  %init = linalg.fill ins(%zero : f32) outs(%out : tensor<16x16x8x8xf32>) -> tensor<16x16x8x8xf32>
  %mm = linalg.mmt4d ins(%a, %b : tensor<16x128x8x1xf32>, tensor<16x128x8x1xf32>) outs(%init : tensor<16x16x8x8xf32>) -> tensor<16x16x8x8xf32>
  %empty = tensor.empty() : tensor<16x16x8x8xf32>
  %sum = linalg.generic {
    indexing_maps = [affine_map<(m,n,i,j)->(m,n,i,j)>, affine_map<(m,n,i,j)->(m,n,i,j)>, affine_map<(m,n,i,j)->(m,n,i,j)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    ins(%bias, %mm : tensor<16x16x8x8xf32>, tensor<16x16x8x8xf32>) outs(%empty : tensor<16x16x8x8xf32>) {
  ^bb0(%bias_value: f32, %acc: f32, %unused: f32):
    %v = math.erf %acc : f32
    linalg.yield %v : f32
  } -> tensor<16x16x8x8xf32>
  return %sum : tensor<16x16x8x8xf32>
}

// CHECK-LABEL: func.func @unsupported_dynamic_shape(
// CHECK: linalg.mmt4d
// CHECK: linalg.generic
// CHECK-NOT: flow.dispatch.workgroups
// CHECK: return

func.func @unsupported_dynamic_shape(%a: tensor<?x128x8x1xf32>, %b: tensor<?x128x8x1xf32>, %bias: tensor<?x16x8x8xf32>, %out: tensor<?x16x8x8xf32>, %other: tensor<?x16x8x8xf32>) -> tensor<?x16x8x8xf32> {
  %zero = arith.constant 0.0 : f32
  %init = linalg.fill ins(%zero : f32) outs(%out : tensor<?x16x8x8xf32>) -> tensor<?x16x8x8xf32>
  %mm = linalg.mmt4d ins(%a, %b : tensor<?x128x8x1xf32>, tensor<?x128x8x1xf32>) outs(%init : tensor<?x16x8x8xf32>) -> tensor<?x16x8x8xf32>
  %sum = linalg.generic {
    indexing_maps = [affine_map<(m,n,i,j)->(m,n,i,j)>, affine_map<(m,n,i,j)->(m,n,i,j)>, affine_map<(m,n,i,j)->(m,n,i,j)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    ins(%bias, %mm : tensor<?x16x8x8xf32>, tensor<?x16x8x8xf32>) outs(%other : tensor<?x16x8x8xf32>) {
  ^bb0(%bias_value: f32, %acc: f32, %unused: f32):
    %v = arith.addf %acc, %bias_value : f32
    linalg.yield %v : f32
  } -> tensor<?x16x8x8xf32>
  return %sum : tensor<?x16x8x8xf32>
}

// Outlining clones this pre-existing payload. It has no selected schedule and
// is intentionally not an LLVM compute owner; the new cutoff must not claim it.
// CHECK-LABEL: flow.executable private @already_wrapped_dispatch_0
// CHECK-NOT: iree_payload.schedule
// CHECK: iree_payload.region
// CHECK-NEXT: iree_payload.yield
// CHECK-NEXT: } {test.keep}
// CHECK-LABEL: func.func @already_wrapped()
// CHECK: flow.dispatch @already_wrapped_dispatch_0::@already_wrapped_dispatch_0
func.func @already_wrapped() {
  flow.dispatch.workgroups[]() : () -> () = () {
    iree_payload.region {
      iree_payload.yield
    } {test.keep}
    flow.return
  }
  return
}

// Already owned computation is outside global inference matching.
// CHECK-LABEL: module @owned
// CHECK: func.func @keep_owned(
// CHECK: linalg.mmt4d
module @owned attributes {iree_payload.compute} {
func.func @keep_owned(%a: tensor<16x128x8x1xf32>, %b: tensor<16x128x8x1xf32>, %bias: tensor<16x16x8x8xf32>, %out: tensor<16x16x8x8xf32>) -> tensor<16x16x8x8xf32> {
  %zero = arith.constant 0.0 : f32
  %init = linalg.fill ins(%zero : f32) outs(%out : tensor<16x16x8x8xf32>) -> tensor<16x16x8x8xf32>
  %mm = linalg.mmt4d ins(%a, %b : tensor<16x128x8x1xf32>, tensor<16x128x8x1xf32>) outs(%init : tensor<16x16x8x8xf32>) -> tensor<16x16x8x8xf32>
  %empty = tensor.empty() : tensor<16x16x8x8xf32>
  %sum = linalg.generic {
    indexing_maps = [affine_map<(m,n,i,j)->(m,n,i,j)>, affine_map<(m,n,i,j)->(m,n,i,j)>, affine_map<(m,n,i,j)->(m,n,i,j)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    ins(%bias, %mm : tensor<16x16x8x8xf32>, tensor<16x16x8x8xf32>) outs(%empty : tensor<16x16x8x8xf32>) {
  ^bb0(%bias_value: f32, %acc: f32, %unused: f32):
    %v = arith.addf %acc, %bias_value : f32
    linalg.yield %v : f32
  } -> tensor<16x16x8x8xf32>
  return %sum : tensor<16x16x8x8xf32>
}
}
}
