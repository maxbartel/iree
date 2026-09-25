// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/dispatch_scheduling/IR/PayloadDialect.h"
#include "experimental/dispatch_scheduling/IR/PayloadOps.h"
#include "experimental/dispatch_scheduling/LLVMCPU/Passes.h"
#include "iree/compiler/Codegen/LLVMCPU/ConversionExtensions.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Transforms/InliningUtils.h"

namespace mlir::iree_compiler::Experimental {
namespace {
struct ConvertPayloadCallOp : ConvertOpToLLVMPattern<PayloadCallOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult matchAndRewrite(
      PayloadCallOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    if (!isa<FlatSymbolRefAttr>(op.getCalleeAttr()) ||
        !op->hasAttr("iree_payload.inline")) {
      return rewriter.notifyMatchFailure(op,
                                         "expected a finalized payload call");
    }
    auto promoted = getTypeConverter()->promoteOperands(
        op.getLoc(), op->getOperands(), adaptor.getOperands(), rewriter,
        /*useBarePtrCallConv=*/false);
    auto call = LLVM::CallOp::create(
        rewriter, op.getLoc(), TypeRange{},
        cast<FlatSymbolRefAttr>(op.getCalleeAttr()), promoted);
    call->setDiscardableAttrs(op->getDiscardableAttrDictionary());
    rewriter.eraseOp(op);
    return success();
  }
};

// Reconcile imports after both classic and owned declarations use LLVM types.
static LogicalResult reconcilePayloadImports(ModuleOp module) {
  SymbolTable symbols(module);
  SmallVector<LLVM::LLVMFuncOp> imports;
  for (auto function : module.getOps<LLVM::LLVMFuncOp>()) {
    if (function->hasAttr("iree_payload.import")) {
      imports.push_back(function);
    }
  }
  for (auto function : imports) {
    auto name = function->getAttrOfType<StringAttr>("iree_payload.import");
    if (!name || !function.isExternal()) {
      return function.emitError(
          "expected a payload bitcode import declaration");
    }
    Operation* existing = symbols.lookup(name);
    if (existing == function) {
      function->removeAttr("iree_payload.import");
      continue;
    }
    if (existing) {
      auto declaration = dyn_cast<LLVM::LLVMFuncOp>(existing);
      // Compare complete declarations, ignoring only their temporary local
      // name, visibility and import marker. In particular, ABI attributes,
      // parameter types and calling conventions must agree with classic code.
      OwningOpRef<LLVM::LLVMFuncOp> comparable = function.clone();
      comparable->setSymName(name);
      (*comparable)->removeAttr("iree_payload.import");
      SymbolTable::setSymbolVisibility(
          *comparable, SymbolTable::getSymbolVisibility(existing));
      // Parsing LLVM assembly makes the default unnamed_addr explicit, while
      // Func-to-LLVM conversion can leave it absent. They mean the same thing.
      if (declaration &&
          comparable->getUnnamedAddr().value_or(LLVM::UnnamedAddr::None) ==
              declaration.getUnnamedAddr().value_or(LLVM::UnnamedAddr::None)) {
        comparable->setUnnamedAddrAttr(declaration.getUnnamedAddrAttr());
      }
      if (!declaration || !declaration.isExternal() ||
          !OperationEquivalence::isEquivalentTo(
              *comparable, existing,
              OperationEquivalence::Flags::IgnoreLocations)) {
        return function.emitError(
                   "conflicting declaration for payload bitcode "
                   "import '")
               << name.getValue() << "'";
      }
      if (failed(SymbolTable::replaceAllSymbolUses(function, name, module))) {
        return function.emitError("failed to reconcile payload bitcode import");
      }
      symbols.erase(function);
    } else {
      if (failed(symbols.rename(function, name))) {
        return function.emitError(
            "failed to restore payload bitcode link name");
      }
      function->removeAttr("iree_payload.import");
    }
  }
  return success();
}

// Inline the structural ABI invocation deterministically, including CFG bodies.
// Optimizer heuristics must not leave a wrapper call or descriptor temporary.
static LogicalResult inlinePayloadCalls(ModuleOp module) {
  SmallVector<LLVM::CallOp> calls;
  module.walk([&](LLVM::CallOp call) {
    if (call->hasAttr("iree_payload.inline")) {
      calls.push_back(call);
    }
  });
  InlinerInterface interface(module.getContext());
  auto cloneBody = [](OpBuilder& builder, Region* source, Block* inlineBlock,
                      Block* continuation, IRMapping& mapping, bool clone) {
    assert(clone && "payload calls always clone their owned body");
    builder.cloneRegionBefore(*source, *inlineBlock->getParent(),
                              continuation->getIterator(), mapping);
  };
  for (LLVM::CallOp call : calls) {
    auto callee = SymbolTable::lookupNearestSymbolFrom<LLVM::LLVMFuncOp>(
        call, call.getCalleeAttr());
    if (!callee || !callee->hasAttr("iree_payload.abi") ||
        failed(inlineCall(interface, cloneBody,
                          cast<CallOpInterface>(call.getOperation()),
                          cast<CallableOpInterface>(callee.getOperation()),
                          &callee.getBody()))) {
      return call.emitError("failed to inline the payload ABI invocation");
    }
    call.erase();
    if (SymbolTable::symbolKnownUseEmpty(callee, module)) {
      callee.erase();
    }
  }
  return success();
}

class PayloadLLVMConversionInterface
    : public LLVMCPUConversionDialectInterface {
 public:
  using LLVMCPUConversionDialectInterface::LLVMCPUConversionDialectInterface;
  void populateConversionPatterns(LLVMTypeConverter& typeConverter,
                                  RewritePatternSet& patterns,
                                  ConversionTarget& target) const override {
    patterns.add<ConvertPayloadCallOp>(typeConverter);
    target.addIllegalOp<PayloadRegionOp, PayloadCallOp>();
  }
  LogicalResult finalizeConversion(ModuleOp module) const override {
    if (failed(reconcilePayloadImports(module))) {
      return failure();
    }
    return inlinePayloadCalls(module);
  }
};
}  // namespace

void registerPayloadLLVMCPUInterfaces(DialectRegistry& registry) {
  LLVM::registerInlinerInterface(registry);
  registry.addExtension(+[](MLIRContext* context, PayloadDialect* dialect) {
    dialect->addInterfaces<PayloadLLVMConversionInterface>();
  });
}
}  // namespace mlir::iree_compiler::Experimental
