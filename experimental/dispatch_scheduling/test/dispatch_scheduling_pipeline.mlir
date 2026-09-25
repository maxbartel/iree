// RUN: iree-compile --iree-plugin=dispatch_scheduling %s --iree-hal-target-backends=llvm-cpu --iree-llvmcpu-target-triple=aarch64-unknown-unknown-eabi-elf --iree-llvmcpu-target-cpu=generic --iree-llvmcpu-enable-ukernels=all --iree-opt-early-data-tiling --iree-dispatch-scheduling --compile-to=dispatch-scheduling -o %t.early
// RUN: FileCheck %s --check-prefix=EARLY --implicit-check-not=flow.dispatch.workgroups --input-file=%t.early
// RUN: iree-compile --iree-plugin=dispatch_scheduling %t.early --compile-from=dispatch-scheduling -o %t.vmfb
// RUN: iree-compile --iree-plugin=dispatch_scheduling %t.early --compile-from=dispatch-scheduling --compile-to=executable-targets -o %t.targets
// RUN: FileCheck %s --check-prefix=TARGETS --input-file=%t.targets --implicit-check-not="iree_payload.region ins(" --implicit-check-not=iree_payload.call --implicit-check-not="llvm.call @entry" --implicit-check-not=memref.copy --implicit-check-not=llvm.intr.memcpy
// RUN: iree-compile --iree-plugin=dispatch_scheduling %s --iree-hal-target-backends=llvm-cpu --iree-llvmcpu-target-triple=aarch64-unknown-unknown-eabi-elf --iree-llvmcpu-target-cpu=generic --iree-llvmcpu-enable-ukernels=all --iree-opt-early-data-tiling --compile-to=dispatch-scheduling | FileCheck %s --check-prefix=OFF

// The body is already LLVM at the scheduling checkpoint. Later lowering
// finishes the bindings and inlines the ABI invocation; it must not copy the
// tensor buffers or leave a wrapper call. CPU feature metadata for the
// microkernel retains the ordinary dispatch ABI.
// TARGETS-LABEL: llvm.func @matmul_dispatch_0(
// TARGETS: llvm.call @iree_uk_mmt4d
// TARGETS: llvm.intr.maximum
// TARGETS-LABEL: llvm.func @odd_dispatch_0(
// TARGETS: llvm.call @iree_uk_mmt4d
// TARGETS: llvm.intr.maximum
// TARGETS-LABEL: llvm.func @dynamic_m_dispatch_0(
// TARGETS: llvm.call @iree_uk_mmt4d
// TARGETS: llvm.fadd

// EARLY-LABEL: flow.executable private @matmul_dispatch_0
// EARLY: iree_payload.region
// EARLY: iree_payload.lowered
// EARLY: llvm.func @entry
// EARLY: llvm.call @iree_uk_mmt4d
// EARLY: llvm.intr.maximum
// EARLY: iree_payload.call
// EARLY-LABEL: util.func public @matmul(
// EARLY: linalg.pack
// EARLY: linalg.pack
// EARLY: flow.dispatch @matmul_dispatch_0::@matmul_dispatch_0
// EARLY: linalg.unpack
// EARLY: util.return
// OFF-LABEL: util.func public @matmul(
// OFF-NOT: flow.dispatch.workgroups
// OFF: linalg.mmt4d
// OFF: linalg.generic
// OFF: util.return

// EARLY-LABEL: flow.executable private @odd_dispatch_0
// EARLY: iree_payload.region
// EARLY: llvm.func @entry
// EARLY: llvm.call @iree_uk_mmt4d
// EARLY: llvm.intr.maximum
// EARLY: iree_payload.call
// EARLY-LABEL: util.func public @odd(
// EARLY: flow.dispatch @odd_dispatch_0::@odd_dispatch_0
// EARLY: linalg.unpack
// EARLY: util.return
// EARLY-LABEL: flow.executable private @dynamic_m_dispatch_0
// EARLY: iree_payload.region
// EARLY: llvm.func @entry
// EARLY: llvm.call @iree_uk_mmt4d
// EARLY: llvm.fadd
// EARLY: iree_payload.call
// EARLY-LABEL: util.func public @dynamic_m(
// EARLY: flow.dispatch @dynamic_m_dispatch_0::@dynamic_m_dispatch_0
// EARLY: linalg.unpack
// EARLY: util.return
// OFF-LABEL: util.func public @odd(
// OFF-NOT: flow.dispatch.workgroups
// OFF: linalg.mmt4d
// OFF: util.return
// OFF-LABEL: util.func public @dynamic_m(
// OFF-NOT: flow.dispatch.workgroups
// OFF: linalg.mmt4d
// OFF: util.return

func.func @matmul(%a: tensor<128x128xf32>, %b: tensor<128x128xf32>) -> tensor<128x128xf32> {
 %zero = arith.constant 0.0 : f32
 %empty = tensor.empty() : tensor<128x128xf32>
 %fill = linalg.fill ins(%zero : f32) outs(%empty : tensor<128x128xf32>) -> tensor<128x128xf32>
 %mm = linalg.matmul ins(%a, %b : tensor<128x128xf32>, tensor<128x128xf32>) outs(%fill : tensor<128x128xf32>) -> tensor<128x128xf32>
 %out = tensor.empty() : tensor<128x128xf32>
 %r = linalg.generic {indexing_maps = [affine_map<(i,j)->(i,j)>,affine_map<(i,j)->(i,j)>], iterator_types=["parallel","parallel"]} ins(%mm : tensor<128x128xf32>) outs(%out : tensor<128x128xf32>) {
 ^bb0(%x: f32, %unused: f32):
  %y = arith.maximumf %x, %zero : f32
  linalg.yield %y : f32
 } -> tensor<128x128xf32>
 return %r : tensor<128x128xf32>
}

func.func @odd(%a: tensor<65x129xf32>, %b: tensor<129x67xf32>) -> tensor<65x67xf32> {
 %zero = arith.constant 0.0 : f32
 %empty = tensor.empty() : tensor<65x67xf32>
 %fill = linalg.fill ins(%zero : f32) outs(%empty : tensor<65x67xf32>) -> tensor<65x67xf32>
 %mm = linalg.matmul ins(%a, %b : tensor<65x129xf32>, tensor<129x67xf32>) outs(%fill : tensor<65x67xf32>) -> tensor<65x67xf32>
 %out = tensor.empty() : tensor<65x67xf32>
 %r = linalg.generic {indexing_maps = [affine_map<(i,j)->(i,j)>,affine_map<(i,j)->(i,j)>], iterator_types=["parallel","parallel"]} ins(%mm : tensor<65x67xf32>) outs(%out : tensor<65x67xf32>) {
 ^bb0(%x: f32, %unused: f32):
  %y = arith.maximumf %x, %zero : f32
  linalg.yield %y : f32
 } -> tensor<65x67xf32>
 return %r : tensor<65x67xf32>
}

func.func @dynamic_m(%a: tensor<?x128xf32>, %b: tensor<128x64xf32>, %bias: tensor<64xf32>) -> tensor<?x64xf32> {
 %zero = arith.constant 0.0 : f32
 %c0 = arith.constant 0 : index
 %m = tensor.dim %a, %c0 : tensor<?x128xf32>
 %empty = tensor.empty(%m) : tensor<?x64xf32>
 %fill = linalg.fill ins(%zero : f32) outs(%empty : tensor<?x64xf32>) -> tensor<?x64xf32>
 %mm = linalg.matmul ins(%a, %b : tensor<?x128xf32>, tensor<128x64xf32>) outs(%fill : tensor<?x64xf32>) -> tensor<?x64xf32>
 %out = tensor.empty(%m) : tensor<?x64xf32>
 %r = linalg.generic {indexing_maps = [affine_map<(i,j)->(i,j)>,affine_map<(i,j)->(j)>,affine_map<(i,j)->(i,j)>], iterator_types=["parallel","parallel"]} ins(%mm, %bias : tensor<?x64xf32>, tensor<64xf32>) outs(%out : tensor<?x64xf32>) {
 ^bb0(%x: f32, %bv: f32, %unused: f32):
  %y = arith.addf %x, %bv : f32
  linalg.yield %y : f32
 } -> tensor<?x64xf32>
 return %r : tensor<?x64xf32>
}
