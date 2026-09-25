// RUN: iree-opt --iree-plugin=dispatch_scheduling %s --iree-dispatch-scheduling-select -o %t
// RUN: FileCheck %s --input-file=%t --implicit-check-not=memref.copy --implicit-check-not=llvm.alloca

// A broadcast bias is the initial accumulator, not a trailing addition.
// It must be written directly into the destination before the microkernel.

// CHECK-LABEL: flow.executable private @broadcast_init_dispatch_0
// CHECK: iree_payload.region
// CHECK: llvm.func @entry
// CHECK: %[[FLAGS:.*]] = llvm.mlir.constant(1793 : i32)
// CHECK: llvm.store
// CHECK: llvm.call @iree_uk_mmt4d({{.*}}, %[[FLAGS]])
// CHECK: llvm.intr.maximum
// CHECK: iree_payload.call
// CHECK-LABEL: func.func @broadcast_init(
// CHECK: flow.dispatch @broadcast_init_dispatch_0::@broadcast_init_dispatch_0
// CHECK: return

#target = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {cpu = "generic", ukernels = "all", cpu_features = "+reserve-x18", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>, max_stack_allocation_size = 32768 : i64, native_vector_size = 16 : i64, target_triple = "aarch64-unknown-unknown-eabi-elf"}>
module attributes {hal.executable.target = #target} {
  func.func @broadcast_init(%a: tensor<16x128x8x1xf32>, %b: tensor<8x128x8x1xf32>, %bias: tensor<8x8xf32>) -> tensor<16x8x8x8xf32> {
    %zero = arith.constant 0.0 : f32
    %empty = tensor.empty() : tensor<16x8x8x8xf32>
    %init = linalg.generic {
      indexing_maps = [affine_map<(m,n,i,j)->(n,j)>, affine_map<(m,n,i,j)->(m,n,i,j)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%bias : tensor<8x8xf32>) outs(%empty : tensor<16x8x8x8xf32>) {
    ^bb0(%value: f32, %unused: f32):
      linalg.yield %value : f32
    } -> tensor<16x8x8x8xf32>
    %mm = linalg.mmt4d ins(%a, %b : tensor<16x128x8x1xf32>, tensor<8x128x8x1xf32>) outs(%init : tensor<16x8x8x8xf32>) -> tensor<16x8x8x8xf32>
    %out = tensor.empty() : tensor<16x8x8x8xf32>
    %relu = linalg.generic {
      indexing_maps = [affine_map<(m,n,i,j)->(m,n,i,j)>, affine_map<(m,n,i,j)->(m,n,i,j)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%mm : tensor<16x8x8x8xf32>) outs(%out : tensor<16x8x8x8xf32>) {
    ^bb0(%value: f32, %unused: f32):
      %r = arith.maximumf %value, %zero : f32
      linalg.yield %r : f32
    } -> tensor<16x8x8x8xf32>
    return %relu : tensor<16x8x8x8xf32>
  }

  // A read/modify/write initializer cannot have its destination replaced by an
  // empty tensor. Keep this unsupported computation on the classic path.
  // CHECK-LABEL: func.func @read_modify_init(
  // CHECK-NOT: flow.dispatch.workgroups
  // CHECK: arith.addf
  // CHECK: linalg.mmt4d
  // CHECK-NOT: flow.dispatch.workgroups
  // CHECK: return
  func.func @read_modify_init(%a: tensor<16x128x8x1xf32>, %b: tensor<8x128x8x1xf32>, %bias: tensor<8x8xf32>, %seed: tensor<16x8x8x8xf32>) -> tensor<16x8x8x8xf32> {
    %zero = arith.constant 0.0 : f32
    %init = linalg.generic {
      indexing_maps = [affine_map<(m,n,i,j)->(n,j)>, affine_map<(m,n,i,j)->(m,n,i,j)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%bias : tensor<8x8xf32>) outs(%seed : tensor<16x8x8x8xf32>) {
    ^bb0(%value: f32, %unused: f32):
      %sum = arith.addf %value, %unused : f32
      linalg.yield %sum : f32
    } -> tensor<16x8x8x8xf32>
    %mm = linalg.mmt4d ins(%a, %b : tensor<16x128x8x1xf32>, tensor<8x128x8x1xf32>) outs(%init : tensor<16x8x8x8xf32>) -> tensor<16x8x8x8xf32>
    %out = tensor.empty() : tensor<16x8x8x8xf32>
    %relu = linalg.generic {
      indexing_maps = [affine_map<(m,n,i,j)->(m,n,i,j)>, affine_map<(m,n,i,j)->(m,n,i,j)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%mm : tensor<16x8x8x8xf32>) outs(%out : tensor<16x8x8x8xf32>) {
    ^bb0(%value: f32, %unused: f32):
      %r = arith.maximumf %value, %zero : f32
      linalg.yield %r : f32
    } -> tensor<16x8x8x8xf32>
    return %relu : tensor<16x8x8x8xf32>
  }
}
