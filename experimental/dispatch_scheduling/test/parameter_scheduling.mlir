// RUN: iree-compile --iree-plugin=dispatch_scheduling %s --iree-hal-target-device=local --iree-hal-local-target-device-backends=llvm-cpu --iree-llvmcpu-target-triple=aarch64-unknown-unknown-eabi-elf --iree-llvmcpu-target-cpu=generic --iree-llvmcpu-enable-ukernels=all --iree-opt-early-data-tiling --iree-dispatch-scheduling --compile-to=dispatch-scheduling -o %t.early
// RUN: FileCheck %s --check-prefix=EARLY --input-file=%t.early
// RUN: iree-compile --iree-plugin=dispatch_scheduling %t.early --compile-from=dispatch-scheduling --iree-parameter-encoder-output-file=%t.encoder.mlir --iree-parameter-encoder-output-scope=packed -o %t.vmfb
// RUN: FileCheck %s --check-prefix=ENCODER --implicit-check-not=iree_payload --input-file=%t.encoder.mlir
// RUN: iree-compile --iree-plugin=dispatch_scheduling %t.encoder.mlir --iree-hal-target-device=local --iree-hal-local-target-device-backends=llvm-cpu --iree-llvmcpu-target-triple=aarch64-unknown-unknown-eabi-elf --iree-llvmcpu-target-cpu=generic -o %t.encoder.vmfb

// Parameter preparation stays on the classic path while inference is already
// LLVM. Both entry points use the same named weight. This test does not require
// deduplicating persistent storage across separately hoisted preparations.

// EARLY: util.initializer
// EARLY-NOT: iree_payload
// EARLY: flow.tensor.constant #flow.parameter.named<"model"::"w">
// EARLY: linalg.pack
// EARLY-NOT: iree_payload
// EARLY: util.return
// EARLY-LABEL: flow.executable private @decode_dispatch_0
// EARLY: iree_payload.region
// EARLY: llvm.func @entry
// EARLY: llvm.call @iree_uk_mmt4d
// EARLY: llvm.intr.maximum
// EARLY: iree_payload.call
// EARLY-LABEL: util.func public @decode(
// EARLY: flow.dispatch @decode_dispatch_0::@decode_dispatch_0
// EARLY: util.return
// EARLY-LABEL: flow.executable private @prefill_dispatch_0
// EARLY: iree_payload.region
// EARLY: llvm.func @entry
// EARLY: llvm.call @iree_uk_mmt4d
// EARLY: llvm.intr.maximum
// EARLY: iree_payload.call
// EARLY-LABEL: util.func public @prefill(
// EARLY: flow.dispatch @prefill_dispatch_0::@prefill_dispatch_0
// EARLY: util.return

// ENCODER: linalg.pack
// ENCODER-LABEL: util.func public @__encode_parameters_all(
// ENCODER: stream.async.parameter.load
// ENCODER: stream.async.dispatch
// ENCODER: stream.async.parameter.scatter

util.global private @weights = #flow.parameter.named<"model"::"w"> : tensor<128x128xf32>

func.func @decode(%a: tensor<1x128xf32>) -> tensor<1x128xf32> {
 %b = util.global.load @weights : tensor<128x128xf32>
 %zero = arith.constant 0.0 : f32
 %empty = tensor.empty() : tensor<1x128xf32>
 %fill = linalg.fill ins(%zero : f32) outs(%empty : tensor<1x128xf32>) -> tensor<1x128xf32>
 %mm = linalg.matmul ins(%a, %b : tensor<1x128xf32>, tensor<128x128xf32>) outs(%fill : tensor<1x128xf32>) -> tensor<1x128xf32>
 %out = tensor.empty() : tensor<1x128xf32>
 %r = linalg.generic {indexing_maps = [affine_map<(i,j)->(i,j)>,affine_map<(i,j)->(i,j)>], iterator_types=["parallel","parallel"]} ins(%mm : tensor<1x128xf32>) outs(%out : tensor<1x128xf32>) {
 ^bb0(%x: f32, %unused: f32):
  %y = arith.maximumf %x, %zero : f32
  linalg.yield %y : f32
 } -> tensor<1x128xf32>
 return %r : tensor<1x128xf32>
}

func.func @prefill(%a: tensor<128x128xf32>) -> tensor<128x128xf32> {
 %b = util.global.load @weights : tensor<128x128xf32>
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
