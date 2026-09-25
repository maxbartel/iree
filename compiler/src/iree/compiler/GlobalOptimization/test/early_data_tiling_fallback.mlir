// RUN: iree-opt --split-input-file --iree-global-opt-early-data-tiling %s | FileCheck %s --check-prefix=EARLY --implicit-check-not=iree_encoding.materialized_layout_target --implicit-check-not=linalg.pack --implicit-check-not=iree_encoding.set_encoding
// RUN: iree-opt --split-input-file --iree-global-opt-early-data-tiling --iree-dispatch-creation-assign-data-tiling-encodings %s | FileCheck %s --check-prefix=LATE

// Unsupported modules must skip early rewriting and still assign late encodings.
module attributes {stream.affinity.default = #hal.device.affinity<@device>} {
  util.global private @device = #hal.device.target<"local", [#hal.executable.target<"vmvx", "vmvx-bytecode-fb">]> : !hal.device
  // EARLY-LABEL: util.func public @vmvx
  // EARLY: linalg.matmul
  // EARLY: arith.maximumf
  // LATE-LABEL: util.func public @vmvx
  // LATE: iree_encoding.set_encoding
  // LATE: linalg.matmul
  // LATE-SAME: tensor<64x128xf32, #{{.*}}encoding
  // LATE: iree_encoding.unset_encoding
  util.func public @vmvx(%lhs: tensor<64x128xf32>, %rhs: tensor<128x64xf32>) -> tensor<64x64xf32> {
    %zero = arith.constant 0.0 : f32
    %empty = tensor.empty() : tensor<64x64xf32>
    %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<64x64xf32>) -> tensor<64x64xf32>
    %mm = linalg.matmul ins(%lhs, %rhs : tensor<64x128xf32>, tensor<128x64xf32>) outs(%init : tensor<64x64xf32>) -> tensor<64x64xf32>
    %relu = linalg.generic {
      indexing_maps = [affine_map<(m, n) -> (m, n)>, affine_map<(m, n) -> (m, n)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%mm : tensor<64x64xf32>) outs(%empty : tensor<64x64xf32>) {
    ^bb0(%value: f32, %unused: f32):
      %result = arith.maximumf %value, %zero : f32
      linalg.yield %result : f32
    } -> tensor<64x64xf32>
    util.return %relu : tensor<64x64xf32>
  }
}

// -----

module attributes {stream.affinity.default = #hal.device.affinity<@device>} {
  util.global private @device = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {target_triple = "aarch64-unknown-unknown-eabi-elf", cpu_features = "+neon", native_vector_size = 16 : i64, iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>, #hal.executable.target<"vmvx", "vmvx-bytecode-fb">]> : !hal.device
  // EARLY-LABEL: util.func public @multiple_targets
  // EARLY: linalg.matmul
  // EARLY: arith.maximumf
  // LATE-LABEL: util.func public @multiple_targets
  // LATE: iree_encoding.set_encoding
  // LATE: linalg.matmul
  // LATE-SAME: tensor<64x128xf32, #{{.*}}encoding
  // LATE: iree_encoding.unset_encoding
  util.func public @multiple_targets(%lhs: tensor<64x128xf32>, %rhs: tensor<128x64xf32>) -> tensor<64x64xf32> {
    %zero = arith.constant 0.0 : f32
    %empty = tensor.empty() : tensor<64x64xf32>
    %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<64x64xf32>) -> tensor<64x64xf32>
    %mm = linalg.matmul ins(%lhs, %rhs : tensor<64x128xf32>, tensor<128x64xf32>) outs(%init : tensor<64x64xf32>) -> tensor<64x64xf32>
    %relu = linalg.generic {
      indexing_maps = [affine_map<(m, n) -> (m, n)>, affine_map<(m, n) -> (m, n)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%mm : tensor<64x64xf32>) outs(%empty : tensor<64x64xf32>) {
    ^bb0(%value: f32, %unused: f32):
      %result = arith.maximumf %value, %zero : f32
      linalg.yield %result : f32
    } -> tensor<64x64xf32>
    util.return %relu : tensor<64x64xf32>
  }
}

// -----

module attributes {stream.affinity.default = #hal.device.affinity<@device>} {
  util.global private @device = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {target_triple = "aarch64-unknown-unknown-eabi-elf", enable_inner_tiled = true, cpu_features = "+neon", native_vector_size = 16 : i64, iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>]> : !hal.device
  // EARLY-LABEL: util.func public @inner_tiled
  // EARLY: linalg.matmul
  // EARLY: arith.maximumf
  // LATE-LABEL: util.func public @inner_tiled
  // LATE: iree_encoding.set_encoding
  // LATE: linalg.matmul
  // LATE-SAME: tensor<64x128xf32, #{{.*}}encoding
  // LATE: iree_encoding.unset_encoding
  util.func public @inner_tiled(%lhs: tensor<64x128xf32>, %rhs: tensor<128x64xf32>) -> tensor<64x64xf32> {
    %zero = arith.constant 0.0 : f32
    %empty = tensor.empty() : tensor<64x64xf32>
    %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<64x64xf32>) -> tensor<64x64xf32>
    %mm = linalg.matmul ins(%lhs, %rhs : tensor<64x128xf32>, tensor<128x64xf32>) outs(%init : tensor<64x64xf32>) -> tensor<64x64xf32>
    %relu = linalg.generic {
      indexing_maps = [affine_map<(m, n) -> (m, n)>, affine_map<(m, n) -> (m, n)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%mm : tensor<64x64xf32>) outs(%empty : tensor<64x64xf32>) {
    ^bb0(%value: f32, %unused: f32):
      %result = arith.maximumf %value, %zero : f32
      linalg.yield %result : f32
    } -> tensor<64x64xf32>
    util.return %relu : tensor<64x64xf32>
  }
}

// -----

module attributes {stream.affinity.default = #hal.device.affinity<@device>} {
  util.global private @device = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {target_triple = "aarch64-unknown-unknown-eabi-elf", cpu_features = "+neon", native_vector_size = 16 : i64}>]> : !hal.device
  // EARLY-LABEL: util.func public @missing_resolver
  // EARLY: linalg.matmul
  // EARLY: arith.maximumf
  // LATE-LABEL: util.func public @missing_resolver
  // LATE: iree_encoding.set_encoding
  // LATE: linalg.matmul
  // LATE-SAME: tensor<64x128xf32, #{{.*}}encoding
  // LATE: iree_encoding.unset_encoding
  util.func public @missing_resolver(%lhs: tensor<64x128xf32>, %rhs: tensor<128x64xf32>) -> tensor<64x64xf32> {
    %zero = arith.constant 0.0 : f32
    %empty = tensor.empty() : tensor<64x64xf32>
    %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<64x64xf32>) -> tensor<64x64xf32>
    %mm = linalg.matmul ins(%lhs, %rhs : tensor<64x128xf32>, tensor<128x64xf32>) outs(%init : tensor<64x64xf32>) -> tensor<64x64xf32>
    %relu = linalg.generic {
      indexing_maps = [affine_map<(m, n) -> (m, n)>, affine_map<(m, n) -> (m, n)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%mm : tensor<64x64xf32>) outs(%empty : tensor<64x64xf32>) {
    ^bb0(%value: f32, %unused: f32):
      %result = arith.maximumf %value, %zero : f32
      linalg.yield %result : f32
    } -> tensor<64x64xf32>
    util.return %relu : tensor<64x64xf32>
  }
}

// -----

module attributes {stream.affinity.default = #hal.device.affinity<@device>} {
  util.global private @device = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {target_triple = "riscv64-unknown-unknown-elf", cpu_features = "+neon", native_vector_size = 16 : i64, iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>]> : !hal.device
  // EARLY-LABEL: util.func public @unsupported_arch
  // EARLY: linalg.matmul
  // EARLY: arith.maximumf
  // LATE-LABEL: util.func public @unsupported_arch
  // LATE: iree_encoding.set_encoding
  // LATE: linalg.matmul
  // LATE-SAME: tensor<64x128xf32, #{{.*}}encoding
  // LATE: iree_encoding.unset_encoding
  util.func public @unsupported_arch(%lhs: tensor<64x128xf32>, %rhs: tensor<128x64xf32>) -> tensor<64x64xf32> {
    %zero = arith.constant 0.0 : f32
    %empty = tensor.empty() : tensor<64x64xf32>
    %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<64x64xf32>) -> tensor<64x64xf32>
    %mm = linalg.matmul ins(%lhs, %rhs : tensor<64x128xf32>, tensor<128x64xf32>) outs(%init : tensor<64x64xf32>) -> tensor<64x64xf32>
    %relu = linalg.generic {
      indexing_maps = [affine_map<(m, n) -> (m, n)>, affine_map<(m, n) -> (m, n)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%mm : tensor<64x64xf32>) outs(%empty : tensor<64x64xf32>) {
    ^bb0(%value: f32, %unused: f32):
      %result = arith.maximumf %value, %zero : f32
      linalg.yield %result : f32
    } -> tensor<64x64xf32>
    util.return %relu : tensor<64x64xf32>
  }
}
