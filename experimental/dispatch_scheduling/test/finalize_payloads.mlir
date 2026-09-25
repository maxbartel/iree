// RUN: iree-opt --iree-plugin=dispatch_scheduling %s --transform-interpreter --iree-transform-dialect-drop-schedule --iree-llvmcpu-finalize-payloads --iree-convert-to-llvm --reconcile-unrealized-casts --canonicalize | FileCheck %s --implicit-check-not=iree_payload --implicit-check-not=unrealized_conversion_cast --implicit-check-not=memref.copy --implicit-check-not=llvm.alloca --implicit-check-not=llvm.noalias --implicit-check-not="llvm.call @entry"

// Structural calls must inline even CFG bodies, while algorithmic helper calls
// retain their own symbols. Both payloads deliberately own the same names; the
// executable already has an unrelated @entry and @factor. The second payload
// reads and writes the same output view, without introducing a noalias promise.
// CHECK: llvm.mlir.global private constant @factor(1.700000e+01
// CHECK: llvm.func @entry()
// CHECK-LABEL: llvm.func @views(
// CHECK-SAME: %[[IN:[a-zA-Z0-9_]+]]: !llvm.ptr, %[[IN_BASE:[a-zA-Z0-9_]+]]: !llvm.ptr, %[[IN_OFF:[a-zA-Z0-9_]+]]: i64, %{{[a-zA-Z0-9_]+}}: i64, %[[IN_STRIDE:[a-zA-Z0-9_]+]]: i64, %{{[a-zA-Z0-9_]+}}: !llvm.ptr, %[[OUT_BASE:[a-zA-Z0-9_]+]]: !llvm.ptr, %[[OUT_OFF:[a-zA-Z0-9_]+]]: i64, %{{[a-zA-Z0-9_]+}}: i64, %[[OUT_STRIDE:[a-zA-Z0-9_]+]]: i64)
// CHECK: llvm.cond_br
// CHECK: llvm.getelementptr %[[IN_BASE]][%[[IN_OFF]]]
// CHECK: llvm.mul %{{.*}}, %[[IN_STRIDE]]
// CHECK: llvm.load
// CHECK: llvm.call @[[FIRST:scale_[0-9]+]]
// CHECK: llvm.getelementptr %[[OUT_BASE]][%[[OUT_OFF]]]
// CHECK: llvm.mul %{{.*}}, %[[OUT_STRIDE]]
// CHECK: llvm.store
// CHECK: llvm.cond_br
// CHECK: llvm.getelementptr %[[OUT_BASE]][%[[OUT_OFF]]]
// CHECK: llvm.mul %{{.*}}, %[[OUT_STRIDE]]
// CHECK: llvm.load
// CHECK: llvm.call @[[SECOND:scale_[0-9]+]]
// CHECK: llvm.store
// CHECK: llvm.return
// CHECK: llvm.mlir.global private @[[GLOBAL_FIRST:factor_[0-9]+]](dense<2.000000e+00>
// CHECK: llvm.func internal @[[FIRST]]
// CHECK: llvm.mlir.addressof @[[GLOBAL_FIRST]]
// CHECK: llvm.mlir.global private @[[GLOBAL_SECOND:factor_[0-9]+]](dense<3.000000e+00>
// CHECK: llvm.func internal @[[SECOND]]
// CHECK: llvm.mlir.addressof @[[GLOBAL_SECOND]]

#target = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {cpu = "generic", cpu_features = "+reserve-x18", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>, max_stack_allocation_size = 32768 : i64, native_vector_size = 16 : i64, target_triple = "aarch64-unknown-unknown-eabi-elf"}>
!view = memref<4xf32, strided<[?], offset: ?>>
module attributes {hal.executable.target = #target} {
  llvm.mlir.global private constant @factor(17.0 : f32) : f32
  func.func private @entry() { return }
  func.func private @views(%in_buffer: !view, %out_buffer: !view) {
      iree_payload.region ins(%in_buffer : !view) outs(%out_buffer : !view) {
      ^bb0(%src: !view, %dst: !view):
        memref.global "private" @factor : memref<1xf32> = dense<2.0>
        func.func private @scale(%value: f32) -> f32 {
          %c0 = arith.constant 0 : index
          %buffer = memref.get_global @factor : memref<1xf32>
          %factor = memref.load %buffer[%c0] : memref<1xf32>
          %scaled = arith.mulf %value, %factor : f32
          return %scaled : f32
        }
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c4 = arith.constant 4 : index
        scf.for %i = %c0 to %c4 step %c1 {
          %value = memref.load %src[%i] : !view
          %scaled = func.call @scale(%value) : (f32) -> f32
          memref.store %scaled, %dst[%i] : !view
        }
        iree_payload.yield
      }
      iree_payload.region ins(%out_buffer : !view) outs(%out_buffer : !view) {
      ^bb0(%src: !view, %dst: !view):
        memref.global "private" @factor : memref<1xf32> = dense<3.0>
        func.func private @scale(%value: f32) -> f32 {
          %c0 = arith.constant 0 : index
          %buffer = memref.get_global @factor : memref<1xf32>
          %factor = memref.load %buffer[%c0] : memref<1xf32>
          %scaled = arith.mulf %value, %factor : f32
          return %scaled : f32
        }
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c4 = arith.constant 4 : index
        scf.for %i = %c0 to %c4 step %c1 {
          %value = memref.load %src[%i] : !view
          %scaled = func.call @scale(%value) : (f32) -> f32
          memref.store %scaled, %dst[%i] : !view
        }
        iree_payload.yield
      }
    return
  }
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %p = transform.structured.match ops{["iree_payload.region"]} in %root : (!transform.any_op) -> !transform.any_op
    %m = transform.iree.outline_payload_region %p : (!transform.any_op) -> !transform.any_op
    %lowered0 = transform.apply_registered_pass "iree-llvmcpu-prepare-payload-codegen" to %m : (!transform.any_op) -> !transform.any_op
    %lowered1 = transform.apply_registered_pass "expand-strided-metadata" to %lowered0 : (!transform.any_op) -> !transform.any_op
    %lowered2 = transform.apply_registered_pass "lower-affine" to %lowered1 : (!transform.any_op) -> !transform.any_op
    %lowered3 = transform.apply_registered_pass "convert-scf-to-cf" to %lowered2 : (!transform.any_op) -> !transform.any_op
    %lowered4 = transform.apply_registered_pass "finalize-memref-to-llvm" to %lowered3 : (!transform.any_op) -> !transform.any_op
    %lowered5 = transform.apply_registered_pass "convert-arith-to-llvm" to %lowered4 : (!transform.any_op) -> !transform.any_op
    %lowered6 = transform.apply_registered_pass "convert-func-to-llvm" to %lowered5 : (!transform.any_op) -> !transform.any_op
    %lowered7 = transform.apply_registered_pass "convert-cf-to-llvm" to %lowered6 : (!transform.any_op) -> !transform.any_op
    %lowered8 = transform.apply_registered_pass "reconcile-unrealized-casts" to %lowered7 : (!transform.any_op) -> !transform.any_op
    %lowered9 = transform.apply_registered_pass "iree-llvmcpu-verify-payload-codegen" to %lowered8 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

}
