// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/dispatch_scheduling/LLVMCPU/Passes.h"
#include "iree/compiler/Codegen/Utils/CodegenOptions.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/Support/Error.h"
#include "llvm/TargetParser/Triple.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

namespace mlir::iree_compiler::Experimental {
#define GEN_PASS_DEF_LLVMCPUPREPAREPAYLOADCODEGENPASS
#define GEN_PASS_DEF_LLVMCPUVERIFYPAYLOADCODEGENPASS
#include "experimental/dispatch_scheduling/LLVMCPU/Passes.h.inc"

namespace {

// Trace the provenance of a written address, not whether two buffers could
// alias physically. Writes through an output argument are legal even if it
// aliases an input; this must never imply noalias on the converted signature.
static bool isWritableAddress(Value value,
                              const DenseMap<Value, bool>& arguments,
                              DenseSet<Value>& visited) {
  if (auto it = arguments.find(value); it != arguments.end()) {
    return it->second;
  }
  if (!visited.insert(value).second) {
    return false;
  }
  if (value.getDefiningOp<memref::AllocaOp>()) {
    return true;
  }
  if (auto view = value.getDefiningOp<ViewLikeOpInterface>()) {
    return isWritableAddress(view.getViewSource(), arguments, visited);
  }
  if (auto cast = value.getDefiningOp<memref::CastOp>()) {
    return isWritableAddress(cast.getSource(), arguments, visited);
  }
  if (auto arg = dyn_cast<BlockArgument>(value)) {
    if (auto loop = dyn_cast<scf::ForOp>(arg.getOwner()->getParentOp())) {
      if (arg.getArgNumber() != 0) {
        unsigned index = arg.getArgNumber() - 1;
        if (!isWritableAddress(loop.getInitArgs()[index], arguments, visited)) {
          return false;
        }
        // The next iteration may receive a different buffer. Checking only the
        // initial value would incorrectly allow an output-to-input swap.
        Value yielded = cast<scf::YieldOp>(loop.getBody()->getTerminator())
                            .getOperand(index);
        return yielded == arg || isWritableAddress(yielded, arguments, visited);
      }
    }
  }
  return false;
}

static bool isWritableAddress(Value value,
                              const DenseMap<Value, bool>& arguments) {
  DenseSet<Value> visited;
  return isWritableAddress(value, arguments, visited);
}

static LogicalResult verifyFunctionEffects(
    func::FuncOp function, const DenseMap<Value, bool>& arguments,
    DenseSet<Operation*>& callStack) {
  for (unsigned index = 0; index < function.getNumArguments(); ++index) {
    if (function.getArgAttr(index, "llvm.noalias")) {
      return function.emitError("payload operands do not imply noalias");
    }
  }
  if (!callStack.insert(function).second) {
    return function.emitError("recursive payload calls are not supported");
  }
  auto result = function.walk([&](Operation* op) -> WalkResult {
    if (op == function.getOperation()) {
      return WalkResult::advance();
    }
    if (auto call = dyn_cast<func::CallOp>(op)) {
      auto callee = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
          call, call.getCalleeAttr());
      if (!callee || callee.isExternal()) {
        call.emitError(
            "payload requires a defined, owned helper; imports need "
            "dispatch ABI support");
        return WalkResult::interrupt();
      }
      DenseMap<Value, bool> calleeArguments;
      for (auto [actual, formal] :
           llvm::zip_equal(call.getOperands(), callee.getArguments())) {
        calleeArguments[formal] = isWritableAddress(actual, arguments);
      }
      return failed(verifyFunctionEffects(callee, calleeArguments, callStack))
                 ? WalkResult::interrupt()
                 : WalkResult::advance();
    }
    auto effects = dyn_cast<MemoryEffectOpInterface>(op);
    if (!effects) {
      if (op->hasTrait<OpTrait::HasRecursiveMemoryEffects>()) {
        return WalkResult::advance();
      }
      op->emitError("cannot prove the payload input/output effect contract");
      return WalkResult::interrupt();
    }
    SmallVector<MemoryEffects::EffectInstance> instances;
    effects.getEffects(instances);
    for (const auto& effect : instances) {
      if (isa<MemoryEffects::Read>(effect.getEffect())) {
        continue;
      }
      if (isa<MemoryEffects::Allocate>(effect.getEffect()) &&
          isa<memref::AllocaOp>(op)) {
        continue;
      }
      if (isa<MemoryEffects::Write>(effect.getEffect()) && effect.getValue() &&
          isWritableAddress(effect.getValue(), arguments)) {
        continue;
      }
      op->emitError(
          "payload may write only through destinations or local stack "
          "storage; heap allocation and undeclared effects require "
          "dispatch ABI support");
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  callStack.erase(function);
  return failure(result.wasInterrupted());
}

static bool isSupportedPayloadImport(LLVM::LLVMFuncOp function,
                                     unsigned indexBitWidth) {
  MLIRContext* context = function.getContext();
  auto bitcode = function->getAttrOfType<BoolAttr>("hal.import.bitcode");
  auto fields = function->getAttrOfType<ArrayAttr>("hal.import.fields");
  if (function.getSymName() != "iree_uk_mmt4d" || !bitcode ||
      !bitcode.getValue() || !fields || fields.size() != 1 ||
      fields[0] != StringAttr::get(context, "processor_data") ||
      function->hasAttr("hal.import.cconv")) {
    return false;
  }
  Type pointer = LLVM::LLVMPointerType::get(context);
  Type index = IntegerType::get(context, indexBitWidth);
  Type i32 = IntegerType::get(context, 32);
  SmallVector<Type> parameters;
  for (unsigned operand = 0; operand < 3; ++operand) {
    parameters.append({pointer, index, index});
  }
  parameters.append(3, index);
  parameters.append(4, i32);
  return function.getFunctionType() ==
         LLVM::LLVMFunctionType::get(i32, parameters);
}

static LogicalResult verifyLLVMClosure(ModuleOp module, StringRef dataLayout) {
  unsigned indexBitWidth = llvm::DataLayout(dataLayout).getPointerSizeInBits();
  if (module
          .walk([&](Operation* op) -> WalkResult {
            if (isa<ModuleOp>(op) ||
                op->getName().getDialectNamespace() == "llvm") {
              return WalkResult::advance();
            }
            op->emitError("non-LLVM operation remains in owned compute body");
            return WalkResult::interrupt();
          })
          .wasInterrupted()) {
    return failure();
  }
  if (module
          .walk([&](Operation* scope) -> WalkResult {
            if (!scope->hasTrait<OpTrait::SymbolTable>()) {
              return WalkResult::advance();
            }
            SmallVector<SymbolTable::SymbolUse> symbolUses;
            // The operation overload deliberately stops at a symbol table.
            // Inspect its regions to verify references in the actual body;
            // nested symbol tables are visited separately by this walk.
            for (Region& region : scope->getRegions()) {
              auto uses = SymbolTable::getSymbolUses(&region);
              if (!uses) {
                scope->emitError("cannot verify owned compute symbol closure");
                return WalkResult::interrupt();
              }
              llvm::append_range(symbolUses, *uses);
            }
            for (const auto& use : symbolUses) {
              Operation* symbol = SymbolTable::lookupNearestSymbolFrom(
                  use.getUser(), use.getSymbolRef());
              if (!symbol || !module->isAncestor(symbol)) {
                use.getUser()->emitError(
                    "compute symbol is not owned by payload");
                return WalkResult::interrupt();
              }
              if (auto global = dyn_cast<LLVM::GlobalOp>(symbol);
                  global && !global.getValueOrNull() &&
                  global.getInitializerRegion().empty()) {
                global.emitError(
                    "external payload globals require explicit "
                    "import support");
                return WalkResult::interrupt();
              }
              if (auto function = dyn_cast<LLVM::LLVMFuncOp>(symbol)) {
                if (function.isExternal()) {
                  // Preserve the existing mmt4d bitcode import until the entry
                  // is inlined into a real dispatch. Its processor-data operand
                  // is then supplied by the normal HAL ABI rewrite.
                  auto caller =
                      use.getUser()->getParentOfType<LLVM::LLVMFuncOp>();
                  if (!isSupportedPayloadImport(function, indexBitWidth)) {
                    function.emitError(
                        "external payload declarations require "
                        "explicit import support");
                    return WalkResult::interrupt();
                  }
                  if (!isa<LLVM::CallOp>(use.getUser()) || !caller ||
                      !caller->hasAttr("iree_payload.abi")) {
                    use.getUser()->emitError(
                        "payload bitcode imports must be called directly "
                        "from the designated compute entry");
                    return WalkResult::interrupt();
                  }
                }
              }
            }
            return WalkResult::advance();
          })
          .wasInterrupted()) {
    return failure();
  }
  LowerToLLVMOptions options(module.getContext());
  options.dataLayout = llvm::DataLayout(dataLayout);
  options.overrideIndexBitwidth(options.dataLayout.getPointerSizeInBits());
  LLVMTypeConverter converter(module.getContext(), options);
  unsigned entryCount = 0;
  for (auto function : module.getOps<LLVM::LLVMFuncOp>()) {
    auto abiAttr = function->getAttrOfType<TypeAttr>("iree_payload.abi");
    if (!abiAttr) {
      continue;
    }
    ++entryCount;
    auto abi = dyn_cast<FunctionType>(abiAttr.getValue());
    if (!abi) {
      return function.emitError("missing original payload buffer ABI");
    }
    TypeConverter::SignatureConversion conversion(abi.getNumInputs());
    Type expected = converter.convertFunctionSignature(
        abi, /*isVariadic=*/false, /*useBarePtrCallConv=*/false, conversion);
    if (expected != function.getFunctionType()) {
      return function.emitError(
          "converted payload entry does not match buffer ABI");
    }
  }
  if (entryCount != 1) {
    return module.emitError("expected exactly one designated compute entry");
  }
  return success();
}

static FailureOr<IREE::HAL::ExecutableTargetAttr> freezePayloadTarget(
    ModuleOp module) {
  auto target = IREE::HAL::ExecutableTargetAttr::lookup(module);
  if (!module->hasAttr("iree_payload.compute") || !target ||
      target.getBackend().getValue() != "llvm-cpu") {
    module.emitError(
        "expected an owned payload with an explicit LLVM CPU target");
    return failure();
  }
  auto layout = getConfigDataLayout(target.getConfiguration());
  auto triple = getConfigTargetTriple(target.getConfiguration());
  if (!layout || layout->empty() || !triple || triple->empty() ||
      !(llvm::Triple(*triple).isAArch64() || llvm::Triple(*triple).isX86())) {
    module.emitError(
        "payload requires a resolved AArch64/x86 triple and data layout");
    return failure();
  }
  auto parsedLayout = llvm::DataLayout::parse(*layout);
  if (!parsedLayout) {
    module.emitError("invalid payload data layout: ")
        << llvm::toString(parsedLayout.takeError());
    return failure();
  }
  for (auto [name, expected] :
       {std::pair<StringRef, StringRef>{"llvm.target_triple", *triple},
        {"llvm.data_layout", *layout}}) {
    if (auto attr = module->getAttrOfType<StringAttr>(name)) {
      if (attr.getValue() != expected) {
        module.emitError("payload target conflicts with existing ") << name;
        return failure();
      }
    }
  }
  // Freeze the resolved configuration on this scope before conversion, so
  // outlining/checkpointing cannot silently inherit a different target later.
  if (auto frozenTarget = module->getAttr("iree_payload.target")) {
    if (frozenTarget != target) {
      module.emitError("payload target differs from its frozen configuration");
      return failure();
    }
  }
  module->setAttr("iree_payload.target", target);
  module->setAttr("hal.executable.target", target);
  module->setAttr("llvm.target_triple",
                  StringAttr::get(module.getContext(), *triple));
  module->setAttr("llvm.data_layout",
                  StringAttr::get(module.getContext(), *layout));
  return target;
}

static LogicalResult verifyPayloadBufferContract(ModuleOp module) {
  unsigned entryCount = 0;
  for (auto function : module.getOps<func::FuncOp>()) {
    if (!function->hasAttr("iree_payload.abi")) {
      continue;
    }
    ++entryCount;
    auto abi = function->getAttrOfType<TypeAttr>("iree_payload.abi");
    if (function.isExternal() || !abi ||
        abi.getValue() != function.getFunctionType()) {
      return function.emitError(
          "unconverted payload entry does not match buffer ABI");
    }
    auto inputs =
        function->getAttrOfType<IntegerAttr>("iree_payload.num_inputs");
    if (!inputs || inputs.getInt() < 0 ||
        inputs.getInt() > function.getNumArguments()) {
      function.emitError("missing payload destination/effect contract");
      return failure();
    }
    DenseMap<Value, bool> arguments;
    for (auto argument : function.getArguments()) {
      arguments[argument] = argument.getArgNumber() >= inputs.getInt();
    }
    DenseSet<Operation*> callStack;
    if (failed(verifyFunctionEffects(function, arguments, callStack))) {
      return failure();
    }
  }
  if (entryCount != 1) {
    module.emitError("expected exactly one unconverted compute entry");
    return failure();
  }
  CPUCodegenOptions options = CPUCodegenOptions::FromFlags::get();
  if (options.instrumentMemoryAccesses) {
    module.emitError("payload instrumentation requires dispatch ABI support");
    return failure();
  }
  return success();
}

class LLVMCPUPreparePayloadCodegenPass
    : public impl::LLVMCPUPreparePayloadCodegenPassBase<
          LLVMCPUPreparePayloadCodegenPass> {
 public:
  using Base::Base;
  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (failed(freezePayloadTarget(module)) ||
        failed(verifyPayloadBufferContract(module))) {
      return signalPassFailure();
    }
    module->setAttr("iree_payload.prepared", UnitAttr::get(&getContext()));
  }
};

class LLVMCPUVerifyPayloadCodegenPass
    : public impl::LLVMCPUVerifyPayloadCodegenPassBase<
          LLVMCPUVerifyPayloadCodegenPass> {
 public:
  using Base::Base;
  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (!module->hasAttr("iree_payload.prepared")) {
      module.emitError(
          "payload must be prepared before explicit LLVM conversion");
      return signalPassFailure();
    }
    if (!module->getAttrOfType<IREE::HAL::ExecutableTargetAttr>(
            "iree_payload.target")) {
      module.emitError("prepared payload is missing its frozen target");
      return signalPassFailure();
    }
    auto target = freezePayloadTarget(module);
    if (failed(target) ||
        failed(verifyLLVMClosure(
            module, *getConfigDataLayout(target->getConfiguration())))) {
      return signalPassFailure();
    }
    module->setAttr("iree_payload.lowered", UnitAttr::get(&getContext()));
  }
};

}  // namespace
}  // namespace mlir::iree_compiler::Experimental
