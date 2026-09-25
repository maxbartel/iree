// RUN: iree-opt --iree-plugin=dispatch_scheduling %s --iree-llvmcpu-prepare-payload-codegen --split-input-file --verify-diagnostics

// input write
#target = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {
  target_triple = "aarch64-unknown-unknown-eabi",
  data_layout = "e-m:e-i64:64-i128:128-n32:64-S128",
  cpu = "generic", cpu_features = "+neon"
}>
module @compute attributes {iree_payload.compute, hal.executable.target = #target} {
  func.func @entry(%input: memref<4xf32>, %output: memref<4xf32>) attributes {
    iree_payload.abi = (memref<4xf32>, memref<4xf32>) -> (),
    iree_payload.num_inputs = 1 : i64
  } {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0.0 : f32
    // expected-error@+1 {{payload may write only through destinations or local stack storage}}
    memref.store %zero, %input[%c0] : memref<4xf32>
    return
  }
}


// -----


// input view write
#target = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {
  target_triple = "aarch64-unknown-unknown-eabi",
  data_layout = "e-m:e-i64:64-i128:128-n32:64-S128",
  cpu = "generic", cpu_features = "+neon"
}>
module @compute attributes {iree_payload.compute, hal.executable.target = #target} {
  func.func @entry(%input: memref<4xf32>, %output: memref<4xf32>) attributes {
    iree_payload.abi = (memref<4xf32>, memref<4xf32>) -> (),
    iree_payload.num_inputs = 1 : i64
  } {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0.0 : f32
    %view = memref.cast %input : memref<4xf32> to memref<?xf32>
    // expected-error@+1 {{payload may write only through destinations or local stack storage}}
    memref.store %zero, %view[%c0] : memref<?xf32>
    return
  }
}


// -----


// heap
#target = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {
  target_triple = "aarch64-unknown-unknown-eabi",
  data_layout = "e-m:e-i64:64-i128:128-n32:64-S128",
  cpu = "generic", cpu_features = "+neon"
}>
module @compute attributes {iree_payload.compute, hal.executable.target = #target} {
  func.func @entry(%input: memref<4xf32>, %output: memref<4xf32>) attributes {
    iree_payload.abi = (memref<4xf32>, memref<4xf32>) -> (),
    iree_payload.num_inputs = 1 : i64
  } {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0.0 : f32
    // expected-error@+1 {{heap allocation and undeclared effects require dispatch ABI support}}
    %allocated = memref.alloc() : memref<4xf32>
    return
  }
}


// -----


// noalias
#target = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {
  target_triple = "aarch64-unknown-unknown-eabi",
  data_layout = "e-m:e-i64:64-i128:128-n32:64-S128",
  cpu = "generic", cpu_features = "+neon"
}>
module @compute attributes {iree_payload.compute, hal.executable.target = #target} {
  // expected-error@+1 {{payload operands do not imply noalias}}
  func.func @entry(%input: memref<4xf32> {llvm.noalias}, %output: memref<4xf32>) attributes {
    iree_payload.abi = (memref<4xf32>, memref<4xf32>) -> (),
    iree_payload.num_inputs = 1 : i64
  } {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0.0 : f32
    memref.store %zero, %output[%c0] : memref<4xf32>
    return
  }
}


// -----


// layout conflict
#target = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {
  target_triple = "aarch64-unknown-unknown-eabi",
  data_layout = "e-m:e-i64:64-i128:128-n32:64-S128",
  cpu = "generic", cpu_features = "+neon"
}>
// expected-error@+1 {{payload target conflicts with existing llvm.data_layout}}
module @compute attributes {iree_payload.compute, hal.executable.target = #target, llvm.data_layout = "e"} {
  func.func @entry(%input: memref<4xf32>, %output: memref<4xf32>) attributes {
    iree_payload.abi = (memref<4xf32>, memref<4xf32>) -> (),
    iree_payload.num_inputs = 1 : i64
  } {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0.0 : f32

    return
  }
}


// -----


// missing target
#target = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {
  target_triple = "aarch64-unknown-unknown-eabi",
  data_layout = "e-m:e-i64:64-i128:128-n32:64-S128",
  cpu = "generic", cpu_features = "+neon"
}>
// expected-error@+1 {{expected an owned payload with an explicit LLVM CPU target}}
module @compute attributes {iree_payload.compute} {
  func.func @entry(%input: memref<4xf32>, %output: memref<4xf32>) attributes {
    iree_payload.abi = (memref<4xf32>, memref<4xf32>) -> (),
    iree_payload.num_inputs = 1 : i64
  } {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0.0 : f32

    return
  }
}


// -----


// invalid layout
#target = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {
  target_triple = "aarch64-unknown-unknown-eabi",
  data_layout = "invalid",
  cpu = "generic", cpu_features = "+neon"
}>
// expected-error@+1 {{invalid payload data layout}}
module @compute attributes {iree_payload.compute, hal.executable.target = #target} {
  func.func @entry(%input: memref<4xf32>, %output: memref<4xf32>) attributes {
    iree_payload.abi = (memref<4xf32>, memref<4xf32>) -> (),
    iree_payload.num_inputs = 1 : i64
  } {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0.0 : f32

    return
  }
}


// -----


// missing contract
#target = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {
  target_triple = "aarch64-unknown-unknown-eabi",
  data_layout = "e-m:e-i64:64-i128:128-n32:64-S128",
  cpu = "generic", cpu_features = "+neon"
}>
module @compute attributes {iree_payload.compute, hal.executable.target = #target} {
  // expected-error@+1 {{missing payload destination/effect contract}}
  func.func @entry(%input: memref<4xf32>, %output: memref<4xf32>) attributes {
    iree_payload.abi = (memref<4xf32>, memref<4xf32>) -> ()

  } {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0.0 : f32

    return
  }
}


// -----


// external helper
#target = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {
  target_triple = "aarch64-unknown-unknown-eabi",
  data_layout = "e-m:e-i64:64-i128:128-n32:64-S128",
  cpu = "generic", cpu_features = "+neon"
}>
module @compute attributes {iree_payload.compute, hal.executable.target = #target} {
  func.func private @external()
  func.func @entry(%input: memref<4xf32>, %output: memref<4xf32>) attributes {
    iree_payload.abi = (memref<4xf32>, memref<4xf32>) -> (),
    iree_payload.num_inputs = 1 : i64
  } {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0.0 : f32
    // expected-error@+1 {{payload requires a defined, owned helper}}
    func.call @external() : () -> ()
    return
  }
}


// -----


// helper input write
#target = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {
  target_triple = "aarch64-unknown-unknown-eabi",
  data_layout = "e-m:e-i64:64-i128:128-n32:64-S128",
  cpu = "generic", cpu_features = "+neon"
}>
module @compute attributes {iree_payload.compute, hal.executable.target = #target} {
  func.func private @helper(%buffer: memref<4xf32>) {
    %c0 = arith.constant 0 : index
    %one = arith.constant 1.0 : f32
    // expected-error@+1 {{payload may write only through destinations or local stack storage}}
    memref.store %one, %buffer[%c0] : memref<4xf32>
    return
  }
  func.func @entry(%input: memref<4xf32>, %output: memref<4xf32>) attributes {
    iree_payload.abi = (memref<4xf32>, memref<4xf32>) -> (),
    iree_payload.num_inputs = 1 : i64
  } {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0.0 : f32
    func.call @helper(%input) : (memref<4xf32>) -> ()
    return
  }
}
