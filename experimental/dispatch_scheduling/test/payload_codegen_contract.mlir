// RUN: iree-opt --iree-plugin=dispatch_scheduling %s --iree-llvmcpu-prepare-payload-codegen -o %t
// RUN: FileCheck %s --check-prefix=PREPARED < %t
// RUN: iree-opt --iree-plugin=dispatch_scheduling %t --pass-pipeline='builtin.module(expand-strided-metadata,lower-affine,convert-scf-to-cf,finalize-memref-to-llvm,convert-arith-to-llvm,convert-func-to-llvm,convert-cf-to-llvm,reconcile-unrealized-casts,iree-llvmcpu-verify-payload-codegen,iree-llvmcpu-verify-payload-codegen)' | FileCheck %s --check-prefix=LLVM

// PREPARED: module @compute attributes
// PREPARED-SAME: iree_payload.prepared
// PREPARED-SAME: iree_payload.target =
// PREPARED-SAME: llvm.data_layout = "e-m:e-i64:64-i128:128-n32:64-S128"
// PREPARED-SAME: llvm.target_triple = "aarch64-unknown-unknown-eabi"
// PREPARED: func.func @entry
// PREPARED: scf.for
// PREPARED: memref.load
// PREPARED: arith.addf
// PREPARED: memref.store
// LLVM: module @compute attributes
// LLVM-SAME: iree_payload.lowered
// LLVM-SAME: iree_payload.prepared
// LLVM: llvm.func @entry
// LLVM-SAME: iree_payload.abi
// LLVM: llvm.cond_br
// LLVM: llvm.load
// LLVM: llvm.fadd
// LLVM: llvm.store
// LLVM: llvm.return

// Preparation sets the target and checks the original effect/ABI contract.
// The explicit upstream passes own body lowering; final verification only
// checks the result. Serialize/restart between preparation and conversion.
#target = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {
  target_triple = "aarch64-unknown-unknown-eabi",
  data_layout = "e-m:e-i64:64-i128:128-n32:64-S128",
  cpu = "generic", cpu_features = "+neon"
}>
module @compute attributes {iree_payload.compute, hal.executable.target = #target} {
  func.func @entry(%input: memref<?xf32, strided<[2], offset: ?>>, %n: index,
                   %output: memref<?xf32, strided<[3], offset: ?>>) attributes {
    iree_payload.abi = (memref<?xf32, strided<[2], offset: ?>>, index, memref<?xf32, strided<[3], offset: ?>>) -> (),
    iree_payload.num_inputs = 2 : i64
  } {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %one = arith.constant 1.0 : f32
    scf.for %i = %c0 to %n step %c1 {
      %v = memref.load %input[%i] : memref<?xf32, strided<[2], offset: ?>>
      %r = arith.addf %v, %one : f32
      memref.store %r, %output[%i] : memref<?xf32, strided<[3], offset: ?>>
    }
    return
  }
}
