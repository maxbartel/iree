// RUN: iree-opt --iree-plugin=dispatch_scheduling %s --iree-llvmcpu-verify-payload-codegen --split-input-file --verify-diagnostics

// expected-error @below {{payload must be prepared before explicit LLVM conversion}}
module @compute attributes {iree_payload.compute} {
  llvm.func @entry() { llvm.return }
}

// -----
#target = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {target_triple = "aarch64-unknown-unknown-eabi", data_layout = "e-m:e-i64:64-i128:128-n32:64-S128"}>
module @compute attributes {iree_payload.compute, iree_payload.prepared, iree_payload.target = #target, hal.executable.target = #target} {
  // expected-error @below {{converted payload entry does not match buffer ABI}}
  llvm.func @entry() attributes {iree_payload.abi = (memref<4xf32>) -> ()} {
    llvm.return
  }
}

// -----
#target = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {target_triple = "aarch64-unknown-unknown-eabi", data_layout = "e-m:e-i64:64-i128:128-n32:64-S128"}>
module @compute attributes {iree_payload.compute, iree_payload.prepared, iree_payload.target = #target, hal.executable.target = #target} {
  // expected-error @below {{non-LLVM operation remains in owned compute body}}
  func.func @entry() attributes {iree_payload.abi = () -> ()} { return }
}

// -----
// expected-error @below {{prepared payload is missing its frozen target}}
module @compute attributes {iree_payload.compute, iree_payload.prepared} {
  llvm.func @entry() { llvm.return }
}
