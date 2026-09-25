// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/dispatch_scheduling/IR/PayloadOps.h"
#include "experimental/dispatch_scheduling/LLVMCPU/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/PassManager.h"

namespace mlir::iree_compiler::Experimental {
#define GEN_PASS_DEF_LLVMCPUFINALIZEPAYLOADSPASS
#include "experimental/dispatch_scheduling/LLVMCPU/Passes.h.inc"

namespace {
class LLVMCPUFinalizePayloadsPass
    : public impl::LLVMCPUFinalizePayloadsPassBase<
          LLVMCPUFinalizePayloadsPass> {
 public:
  using Base::Base;
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<IREE::Codegen::IREECodegenDialect>();
    registry.insert<LLVM::LLVMDialect>();
  }
  void runOnOperation() override {
    ModuleOp executable = getOperation();
    auto target = IREE::HAL::ExecutableTargetAttr::lookup(executable);
    SmallVector<PayloadRegionOp> payloads;
    executable.walk(
        [&](PayloadRegionOp payload) { payloads.push_back(payload); });
    SymbolTable executableSymbols(executable);
    IRRewriter rewriter(&getContext());
    for (auto payload : payloads) {
      Block& body = payload.getBody().front();
      auto modules = body.getOps<ModuleOp>();
      auto calls = body.getOps<PayloadCallOp>();
      if (payload.getNumResults() || !llvm::hasSingleElement(modules) ||
          !llvm::hasSingleElement(calls) || body.getOperations().size() != 3) {
        payload.emitError(
            "expected an outlined buffer payload with one invocation");
        return signalPassFailure();
      }
      ModuleOp compute = *modules.begin();
      PayloadCallOp call = *calls.begin();
      auto frozen = compute->getAttr("iree_payload.target");
      if (!target || frozen != target) {
        payload.emitError("payload target does not match its final executable");
        return signalPassFailure();
      }
      // Revalidate serialized LLVM and its original buffer ABI immediately
      // before removing ownership. This does not schedule or lower numerical IR
      // again.
      OpPassManager verifier(ModuleOp::getOperationName());
      verifier.addPass(createLLVMCPUVerifyPayloadCodegenPass());
      if (failed(runPipeline(verifier, compute))) {
        return signalPassFailure();
      }
      auto entry = SymbolTable::lookupNearestSymbolFrom<LLVM::LLVMFuncOp>(
          call, call.getCalleeAttr());
      if (!entry) {
        call.emitError("missing lowered compute entry");
        return signalPassFailure();
      }
      SmallVector<Operation*> symbols;
      for (Operation& op : *compute.getBody()) {
        if (!isa<LLVM::LLVMFuncOp, LLVM::GlobalOp>(op)) {
          op.emitError(
              "unsupported owned symbol during CPU payload finalization");
          return signalPassFailure();
        }
        symbols.push_back(&op);
      }
      SymbolTable computeSymbols(compute);
      for (Operation* symbol : symbols) {
        if (auto function = dyn_cast<LLVM::LLVMFuncOp>(symbol);
            function && function.isExternal()) {
          // Keep the link name while reconciling the owned symbol namespace.
          // Equivalent classic declarations may still be func.func here;
          // coalesce them once all declarations have LLVM signatures.
          function->setAttr("iree_payload.import", function.getSymNameAttr());
        }
        if (failed(
                computeSymbols.renameToUnique(symbol, {&executableSymbols}))) {
          symbol->emitError("failed to reconcile payload symbol names");
          return signalPassFailure();
        }
      }
      for (Operation* symbol : symbols) {
        computeSymbols.remove(symbol);
        rewriter.moveOpBefore(symbol, executable.getBody(),
                              executable.getBody()->end());
        executableSymbols.insert(symbol);
        SymbolTable::setSymbolVisibility(symbol,
                                         SymbolTable::Visibility::Private);
        if (auto function = dyn_cast<LLVM::LLVMFuncOp>(symbol);
            function && !function.isExternal()) {
          function.setLinkage(LLVM::Linkage::Internal);
        }
      }
      IRMapping mapping;
      for (auto [argument, operand] :
           llvm::zip_equal(body.getArguments(), payload.getOperands())) {
        mapping.map(argument, operand);
      }
      SmallVector<Value> arguments;
      for (Value argument : call.getArguments()) {
        arguments.push_back(mapping.lookupOrDefault(argument));
      }
      rewriter.setInsertionPoint(payload);
      auto invocation = PayloadCallOp::create(
          rewriter, payload.getLoc(),
          FlatSymbolRefAttr::get(entry.getSymNameAttr()), arguments);
      invocation->setAttr("iree_payload.inline", rewriter.getUnitAttr());
      rewriter.eraseOp(payload);
    }
  }
};
}  // namespace
}  // namespace mlir::iree_compiler::Experimental
