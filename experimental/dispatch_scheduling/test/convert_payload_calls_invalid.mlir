// RUN: iree-opt --iree-plugin=dispatch_scheduling %s --iree-convert-to-llvm --verify-diagnostics

// A wrapper that cannot inline must diagnose instead of silently adding a call.
module {
  llvm.func internal @compute(%arg: !llvm.ptr) attributes {
    no_inline, iree_payload.abi = (!llvm.ptr) -> ()
  } {
    llvm.return
  }
  func.func private @caller(%arg: !llvm.ptr) {
    // expected-error @+1 {{failed to inline the payload ABI invocation}}
    iree_payload.call @compute(%arg) {iree_payload.inline} : (!llvm.ptr) -> ()
    return
  }
}
