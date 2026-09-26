// RUN: iree-compile %s --iree-plugin=dispatch_scheduling --iree-hal-target-backends=llvm-cpu --iree-llvmcpu-target-triple=aarch64-unknown-unknown-eabi-elf --iree-llvmcpu-target-cpu=generic --iree-llvmcpu-enable-ukernels=all --iree-opt-early-data-tiling --iree-dispatch-scheduling --iree-opt-const-eval=true --compile-to=dispatch-scheduling -o %t.evaluated
// RUN: FileCheck %s --check-prefix=EVAL --implicit-check-not=util.initializer --input-file=%t.evaluated
// RUN: iree-compile %s --iree-plugin=dispatch_scheduling --iree-hal-target-backends=llvm-cpu --iree-llvmcpu-target-triple=aarch64-unknown-unknown-eabi-elf --iree-llvmcpu-target-cpu=generic --iree-llvmcpu-enable-ukernels=all --iree-opt-early-data-tiling --iree-dispatch-scheduling --iree-opt-const-eval=false --compile-to=dispatch-scheduling -o %t.runtime
// RUN: FileCheck %s --check-prefix=RUNTIME --input-file=%t.runtime
// RUN: iree-compile --iree-plugin=dispatch_scheduling %t.evaluated --compile-from=dispatch-scheduling -o %t.evaluated.vmfb
// RUN: iree-compile --iree-plugin=dispatch_scheduling %t.runtime --compile-from=dispatch-scheduling -o %t.runtime.vmfb

// Constant weight preparation uses the existing JIT path. With consteval off,
// the initializer remains unscheduled, while both inference bodies are LLVM.
// The nonuniform outer product requires evaluation, not just splat folding.
// EVAL: util.global private @weights = dense<
// EVAL-LABEL: flow.executable private @decode_dispatch_0
// EVAL: llvm.call @iree_uk_mmt4d
// EVAL: llvm.intr.maximum
// EVAL-LABEL: flow.executable private @prefill_dispatch_0
// EVAL: llvm.call @iree_uk_mmt4d
// EVAL: llvm.intr.maximum

// RUNTIME: util.global private @weights : tensor<4x8xf32>
// RUNTIME: util.initializer
// RUNTIME-NOT: iree_payload
// RUNTIME: linalg.generic
// RUNTIME: arith.mulf
// RUNTIME: util.global.store
// RUNTIME-NOT: iree_payload
// RUNTIME: util.return
// RUNTIME: util.initializer
// RUNTIME: linalg.pack
// RUNTIME-LABEL: flow.executable private @decode_dispatch_0
// RUNTIME: llvm.call @iree_uk_mmt4d
// RUNTIME: llvm.intr.maximum
// RUNTIME-LABEL: flow.executable private @prefill_dispatch_0
// RUNTIME: llvm.call @iree_uk_mmt4d
// RUNTIME: llvm.intr.maximum

util.global private @weights : tensor<4x8xf32>

util.initializer {
  %row = arith.constant dense<[1.0, -2.0, 3.0, -4.0]> : tensor<4xf32>
  %col = arith.constant dense<[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]> : tensor<8xf32>
  %empty = tensor.empty() : tensor<4x8xf32>
  %weights = linalg.generic {
      indexing_maps = [affine_map<(i, j) -> (i)>,
                       affine_map<(i, j) -> (j)>,
                       affine_map<(i, j) -> (i, j)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%row, %col : tensor<4xf32>, tensor<8xf32>)
      outs(%empty : tensor<4x8xf32>) {
  ^bb0(%r: f32, %c: f32, %unused: f32):
    %value = arith.mulf %r, %c : f32
    linalg.yield %value : f32
  } -> tensor<4x8xf32>
  util.global.store %weights, @weights : tensor<4x8xf32>
  util.return
}

func.func @decode(%a: tensor<1x4xf32>) -> tensor<1x8xf32> {
 %b = util.global.load @weights : tensor<4x8xf32>
 %zero = arith.constant 0.0 : f32
 %empty = tensor.empty() : tensor<1x8xf32>
 %fill = linalg.fill ins(%zero : f32) outs(%empty : tensor<1x8xf32>) -> tensor<1x8xf32>
 %mm = linalg.matmul ins(%a, %b : tensor<1x4xf32>, tensor<4x8xf32>) outs(%fill : tensor<1x8xf32>) -> tensor<1x8xf32>
 %out = tensor.empty() : tensor<1x8xf32>
 %r = linalg.generic {indexing_maps = [affine_map<(i,j)->(i,j)>,affine_map<(i,j)->(i,j)>], iterator_types=["parallel","parallel"]} ins(%mm : tensor<1x8xf32>) outs(%out : tensor<1x8xf32>) {
 ^bb0(%x: f32, %unused: f32):
  %y = arith.maximumf %x, %zero : f32
  linalg.yield %y : f32
 } -> tensor<1x8xf32>
 return %r : tensor<1x8xf32>
}

func.func @prefill(%a: tensor<128x4xf32>) -> tensor<128x8xf32> {
 %b = util.global.load @weights : tensor<4x8xf32>
 %zero = arith.constant 0.0 : f32
 %empty = tensor.empty() : tensor<128x8xf32>
 %fill = linalg.fill ins(%zero : f32) outs(%empty : tensor<128x8xf32>) -> tensor<128x8xf32>
 %mm = linalg.matmul ins(%a, %b : tensor<128x4xf32>, tensor<4x8xf32>) outs(%fill : tensor<128x8xf32>) -> tensor<128x8xf32>
 %out = tensor.empty() : tensor<128x8xf32>
 %r = linalg.generic {indexing_maps = [affine_map<(i,j)->(i,j)>,affine_map<(i,j)->(i,j)>], iterator_types=["parallel","parallel"]} ins(%mm : tensor<128x8xf32>) outs(%out : tensor<128x8xf32>) {
 ^bb0(%x: f32, %unused: f32):
  %y = arith.maximumf %x, %zero : f32
  linalg.yield %y : f32
 } -> tensor<128x8xf32>
 return %r : tensor<128x8xf32>
}
