// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_EXPERIMENTAL_DISPATCH_SCHEDULING_SCHEDULING_PASSES
#define IREE_EXPERIMENTAL_DISPATCH_SCHEDULING_SCHEDULING_PASSES
include "mlir/Pass/PassBase.td"

def SelectDispatchSchedulesPass : Pass<"iree-dispatch-scheduling-select", "mlir::ModuleOp"> {
  let summary = "Schedule selected global computations before dispatch creation";
  let options = [
    Option<"libraryFileName", "library-file", "std::string", "\"\"",
           "Transform library containing selection and dispatch entry points">,
    Option<"librarySource", "library-source", "std::string", "\"\"",
           "Immutable transform library source captured in reproducers">,
    Option<"stopAfterSelection", "stop-after-selection", "bool", "false",
           "Stop before numerical lowering for a serialized schedule handoff">
  ];
}

def ExecuteDispatchSchedulesPass : Pass<"iree-dispatch-scheduling-execute", "mlir::ModuleOp"> {
  let summary = "Run recorded dispatch schedules in parallel on isolated executables";
  let description = [{
    Resolve every pending executable's library hash, entry point and target
    before starting any numerical transformation. Libraries are loaded from
    immutable source snapshots in the module, without consulting file paths.
    Each executable runs with independent Transform state using the nested pass
    manager's parallel executor. Completed and unselected executables are skipped.
  }];
}

#endif
