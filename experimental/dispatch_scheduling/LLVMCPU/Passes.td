// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include "mlir/Pass/PassBase.td"

def LLVMCPUPreparePayloadCodegenPass :
    Pass<"iree-llvmcpu-prepare-payload-codegen", "ModuleOp"> {
  let summary = "Check the buffer effect contract and freeze the payload target";
  let description = [{
    Prepare an owned compute module for explicit script-controlled LLVM
    conversion. Verify the original buffer ABI and readonly-input/writable-output
    contract while memref operations are still analyzable. Freeze the resolved
    CPU target and set LLVM triple/data layout attributes. This pass schedules
    and lowers no computation. Run it before lowering microkernel ops to calls.
  }];
}

def LLVMCPUVerifyPayloadCodegenPass :
    Pass<"iree-llvmcpu-verify-payload-codegen", "ModuleOp"> {
  let summary = "Verify explicit LLVM conversion and mark the payload complete";
  let description = [{
    Require a previously prepared compute module, verify the frozen CPU target,
    LLVM-only symbol closure, supported imports and converted buffer ABI, then
    mark the body ready for host boundary lowering. This pass performs no
    computation lowering and can be repeated on a completed body.
  }];
}

def LLVMCPUPreparePayloadBoundariesPass :
    Pass<"iree-llvmcpu-prepare-payload-boundaries", "ModuleOp"> {
  let summary = "Resolve dispatch buffer views around fully lowered payloads";
  let description = [{
    Resolve tensor adapters, slices and shape metadata to views of dispatch
    binding buffers. Require the payload computation to be fully lowered and
    reject remaining tensor adapters. An in-place result needs no copy; results
    in private storage still require a write to their dispatch destination.
  }];
}

def LLVMCPUFinalizePayloadsPass :
    Pass<"iree-llvmcpu-finalize-payloads", "ModuleOp"> {
  let summary = "Transfer owned LLVM symbols into their dispatch executable";
  let description = [{
    Revalidate the target, symbol closure and recorded buffer ABI of each lowered
    payload. Reconcile symbol names with the executable, then replace the owned
    wrapper with a marked invocation. CPU conversion lowers that invocation and
    must inline it before adapting imports to the dispatch ABI.
  }];
}
