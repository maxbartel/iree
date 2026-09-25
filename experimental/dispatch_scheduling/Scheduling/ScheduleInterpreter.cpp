// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/dispatch_scheduling/IR/PayloadOps.h"
#include "experimental/dispatch_scheduling/LLVMCPU/Passes.h"
#include "experimental/dispatch_scheduling/Scheduling/Passes.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/Encoding/Utils/Utils.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/Analysis/DeviceAnalysis.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SHA256.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/Transforms/TransformInterpreterUtils.h"
#include "mlir/Support/FileUtilities.h"

namespace mlir::iree_compiler::Experimental {
#define GEN_PASS_DEF_SELECTDISPATCHSCHEDULESPASS
#define GEN_PASS_DEF_EXECUTEDISPATCHSCHEDULESPASS
#include "experimental/dispatch_scheduling/Scheduling/Passes.h.inc"

namespace {
constexpr StringLiteral kSchedule = "iree_payload.schedule";
constexpr StringLiteral kEnvironment = "iree_payload.schedule_environment";
constexpr StringLiteral kCompleted = "iree_payload.schedule_completed";
constexpr StringLiteral kSources = "iree_payload.schedule_sources";
constexpr StringLiteral kPreserved = "iree_payload.preserved";

static std::string digestSource(StringRef source) {
  return llvm::toHex(llvm::SHA256::hash(llvm::arrayRefFromStringRef(source)),
                     /*LowerCase=*/true);
}

struct PreparedSchedule {
  ModuleOp library;
  transform::NamedSequenceOp entry;
  IREE::HAL::ExecutableTargetAttr target;
  StringAttr digest;
  DictionaryAttr configuration;
  DictionaryAttr selection;
};
using PreparedSchedules = DenseMap<Operation*, PreparedSchedule>;

static bool hasScheduleSignature(transform::NamedSequenceOp entry) {
  if (!entry) {
    return false;
  }
  auto type = entry.getFunctionType();
  return type.getNumResults() == 0 &&
         (type.getNumInputs() == 1 || type.getNumInputs() == 2) &&
         isa<transform::TransformHandleTypeInterface>(type.getInput(0)) &&
         (type.getNumInputs() == 1 ||
          isa<transform::TransformParamTypeInterface>(type.getInput(1)));
}

static LogicalResult applySchedule(Operation* payload,
                                   transform::NamedSequenceOp entry,
                                   ModuleOp library, DictionaryAttr parameter) {
  RaggedArray<transform::MappedValue> bindings;
  bindings.push_back(ArrayRef<Operation*>{payload});
  if (entry.getFunctionType().getNumInputs() == 2) {
    bindings.push_back(ArrayRef<Attribute>{parameter});
  }
  transform::TransformOptions options;
  return transform::applyTransformNamedSequence(
      std::move(bindings), entry, library, options.enableExpensiveChecks(true));
}

// Only immutable library IR and dispatch information are shared. Each pass
// clone creates its own TransformState and nested verifier pipelines.
class ExecuteDispatchSchedulePass final
    : public PassWrapper<ExecuteDispatchSchedulePass,
                         OperationPass<IREE::Flow::ExecutableOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ExecuteDispatchSchedulePass)
  explicit ExecuteDispatchSchedulePass(
      std::shared_ptr<const PreparedSchedules> schedules)
      : schedules(std::move(schedules)) {}

  StringRef getArgument() const override {
    return "iree-dispatch-scheduling-execute-one";
  }
  StringRef getDescription() const override {
    return "Interpret the recorded numerical schedule for one executable";
  }
  void getDependentDialects(DialectRegistry& registry) const override {
    registerTransformDialectTranslationDependentDialects(registry);
  }
  void runOnOperation() override {
    auto executable = getOperation();
    auto it = schedules->find(executable);
    if (it == schedules->end()) {
      return;
    }
    const PreparedSchedule& schedule = it->second;
    ModuleOp module = executable.getInnerModule();
    StringAttr exportName = SymbolTable::getSymbolName(
        *executable.getOps<IREE::Flow::ExecutableExportOp>().begin());
    Attribute oldTarget = module->getAttr("hal.executable.target");
    module->setAttr("hal.executable.target", schedule.target);
    auto restoreTarget = llvm::scope_exit([&] {
      // A user sequence may replace the inner module. Never retain a pointer
      // into the rewritten body across interpretation.
      ModuleOp currentModule = executable.getInnerModule();
      if (!currentModule) {
        return;
      }
      if (oldTarget) {
        currentModule->setAttr("hal.executable.target", oldTarget);
      } else {
        currentModule->removeAttr("hal.executable.target");
      }
    });
    if (failed(applySchedule(executable, schedule.entry, schedule.library,
                             schedule.configuration))) {
      executable.emitError(
          "recorded dispatch schedule failed; classic fallback "
          "is not rollback");
      return signalPassFailure();
    }
    module = executable.getInnerModule();
    auto exports = executable.getOps<IREE::Flow::ExecutableExportOp>();
    if (!module || !llvm::hasSingleElement(exports) ||
        SymbolTable::getSymbolName(*exports.begin()) != exportName) {
      executable.emitError(
          "dispatch schedule must retain its export and module boundaries");
      return signalPassFailure();
    }
    SmallVector<PayloadRegionOp> payloads;
    module.walk([&](PayloadRegionOp op) { payloads.push_back(op); });
    if (payloads.empty()) {
      executable.emitError(
          "dispatch schedule must retain its payload boundary");
      return signalPassFailure();
    }
    for (auto payload : payloads) {
      auto owned = payload.getBody().front().getOps<ModuleOp>();
      if (!llvm::hasSingleElement(owned)) {
        payload.emitError(
            "dispatch schedule must produce one owned LLVM compute module");
        return signalPassFailure();
      }
      ModuleOp compute = *owned.begin();
      OpPassManager verifier(ModuleOp::getOperationName());
      verifier.addPass(createLLVMCPUVerifyPayloadCodegenPass());
      if (failed(runPipeline(verifier, compute))) {
        return signalPassFailure();
      }
      compute->setAttr("iree_payload.schedule_sha256", schedule.digest);
    }
    (*exports.begin())->setAttr(kCompleted, schedule.selection);
  }

 private:
  std::shared_ptr<const PreparedSchedules> schedules;
};

class ExecuteDispatchSchedulesPass final
    : public impl::ExecuteDispatchSchedulesPassBase<
          ExecuteDispatchSchedulesPass> {
 public:
  using Base::Base;
  void getDependentDialects(DialectRegistry& registry) const override {
    registerTransformDialectTranslationDependentDialects(registry);
  }
  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto sources = module->getAttrOfType<DictionaryAttr>(kSources);
    auto* dialect =
        getContext().getOrLoadDialect<IREE::Codegen::IREECodegenDialect>();
    auto schedules = std::make_shared<PreparedSchedules>();
    // Preflight ALL jobs before any of them rewrites IR. In particular, no
    // worker parses libraries or populates a shared mutable symbol/cache map.
    for (auto executable : module.getOps<IREE::Flow::ExecutableOp>()) {
      SmallVector<IREE::Flow::ExecutableExportOp> exports(
          executable.getOps<IREE::Flow::ExecutableExportOp>());
      bool hasSelection = llvm::any_of(exports, [](auto op) {
        return op->hasAttr(kSchedule) || op->hasAttr(kEnvironment);
      });
      if (!hasSelection) {
        continue;
      }
      if (exports.size() != 1 || !executable.getInnerModule()) {
        executable.emitError(
            "dispatch scheduling requires one export and one compute module");
        return signalPassFailure();
      }
      auto exportOp = exports.front();
      auto choice = exportOp->getAttrOfType<DictionaryAttr>(kSchedule);
      auto environment = exportOp->getAttrOfType<DictionaryAttr>(kEnvironment);
      auto entryName =
          choice ? choice.getAs<StringAttr>("entry_point") : StringAttr();
      auto config = choice ? choice.getAs<DictionaryAttr>("configuration")
                           : DictionaryAttr();
      auto hash = environment ? environment.getAs<StringAttr>("library_sha256")
                              : StringAttr();
      auto target =
          environment
              ? environment.getAs<IREE::HAL::ExecutableTargetAttr>("target")
              : IREE::HAL::ExecutableTargetAttr();
      if (!entryName || !config || !hash || !target ||
          !isLLVMCPUBackend(target)) {
        exportOp.emitError(
            "malformed dispatch schedule selection or environment");
        return signalPassFailure();
      }
      auto source =
          sources ? sources.getAs<StringAttr>(hash.getValue()) : StringAttr();
      if (!source || digestSource(source.getValue()) != hash.getValue()) {
        exportOp.emitError(
            "missing or mismatched dispatch schedule source snapshot");
        return signalPassFailure();
      }
      if (auto resolved = IREE::HAL::ExecutableTargetAttr::lookup(executable);
          resolved && resolved != target) {
        exportOp.emitError(
            "recorded dispatch schedule target differs from resolved target");
        return signalPassFailure();
      }
      if (Attribute layoutTarget = module->getAttr(
              IREE::Encoding::kMaterializedLayoutTargetAttrName);
          layoutTarget && layoutTarget != target) {
        exportOp.emitError(
            "recorded dispatch schedule target differs from "
            "materialized layout target");
        return signalPassFailure();
      }
      auto selection = DictionaryAttr::get(
          &getContext(),
          {NamedAttribute(StringAttr::get(&getContext(), "schedule"), choice),
           NamedAttribute(StringAttr::get(&getContext(), "environment"),
                          environment)});
      if (Attribute completed = exportOp->getAttr(kCompleted)) {
        if (completed != selection) {
          exportOp.emitError("completed dispatch schedule selection changed");
          return signalPassFailure();
        }
        continue;
      }
      auto parsed = dialect->getOrParseTransformLibraryModule(
          "iree-dispatch-schedule-" + hash.getValue().str(), source.getValue());
      if (failed(parsed)) {
        return signalPassFailure();
      }
      auto entry = dyn_cast_or_null<transform::NamedSequenceOp>(
          SymbolTable::lookupSymbolIn(*parsed, entryName));
      if (!entry) {
        exportOp.emitError("missing recorded dispatch schedule entry: ")
            << entryName.getValue();
        return signalPassFailure();
      }
      if (!hasScheduleSignature(entry)) {
        exportOp.emitError(
            "dispatch schedule entry must take one executable "
            "handle, an optional configuration parameter, and return nothing");
        return signalPassFailure();
      }
      auto type = entry.getFunctionType();
      auto handle =
          cast<transform::TransformHandleTypeInterface>(type.getInput(0));
      if (failed(handle.checkPayload(exportOp.getLoc(), {executable})
                     .checkAndReport())) {
        return signalPassFailure();
      }
      if (type.getNumInputs() == 2) {
        auto parameter =
            cast<transform::TransformParamTypeInterface>(type.getInput(1));
        if (failed(parameter.checkPayload(exportOp.getLoc(), {config})
                       .checkAndReport())) {
          return signalPassFailure();
        }
      }
      schedules->try_emplace(
          executable,
          PreparedSchedule{*parsed, entry, target, hash, config, selection});
    }
    if (schedules->empty()) {
      return;
    }
    OpPassManager workers(ModuleOp::getOperationName());
    workers.addNestedPass<IREE::Flow::ExecutableOp>(
        std::make_unique<ExecuteDispatchSchedulePass>(schedules));
    if (failed(runPipeline(workers, module))) {
      return signalPassFailure();
    }
  }
};

class SelectDispatchSchedulesPass final
    : public impl::SelectDispatchSchedulesPassBase<
          SelectDispatchSchedulesPass> {
 public:
  using Base::Base;
  void getDependentDialects(DialectRegistry& registry) const override {
    registerTransformDialectTranslationDependentDialects(registry);
  }

  LogicalResult initialize(MLIRContext* context) override {
    // Snapshot once, before rewriting anything. The source option is retained
    // in pass reproducers and copies; the cache key includes the content hash,
    // so another compilation in the same context may use a changed file.
    if (librarySource.empty()) {
      if (!libraryFileName.empty()) {
        std::string error;
        auto buffer = mlir::openInputFile(libraryFileName, &error);
        if (!buffer) {
          return emitError(UnknownLoc::get(context))
                 << "cannot load dispatch transform library: " << error;
        }
        librarySource = buffer->getBuffer().str();
      } else {
        return emitError(UnknownLoc::get(context))
               << "expected a Transform library file or source";
      }
    }
    sourceDigest = digestSource(librarySource);
    auto* dialect =
        context->getOrLoadDialect<IREE::Codegen::IREECodegenDialect>();
    auto parsed = dialect->getOrParseTransformLibraryModule(
        "iree-dispatch-schedule-" + sourceDigest, librarySource);
    if (failed(parsed)) {
      return failure();
    }
    library = *parsed;
    auto candidate = transform::detail::findTransformEntryPoint(
        library, library, "__transform_main");
    entry = candidate
                ? dyn_cast<transform::NamedSequenceOp>(candidate.getOperation())
                : transform::NamedSequenceOp();
    if (!entry) {
      return library.emitError(
          "missing dispatch transform entry __transform_main");
    }
    if (!hasScheduleSignature(entry)) {
      return entry.emitError(
          "global dispatch schedule must take one module handle, an optional "
          "environment parameter, and return nothing");
    }
    return success();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    IREE::HAL::DeviceAnalysis analysis(module);
    if (failed(analysis.run())) {
      return signalPassFailure();
    }
    SetVector<IREE::HAL::ExecutableTargetAttr> targets;
    analysis.gatherAllExecutableTargets(targets);
    auto explicitTarget =
        module->getAttrOfType<IREE::HAL::ExecutableTargetAttr>(
            "hal.executable.target");
    if (targets.empty() && explicitTarget) {
      targets.insert(explicitTarget);
    }
    if (Attribute layoutTarget = module->getAttr(
            IREE::Encoding::kMaterializedLayoutTargetAttrName)) {
      if (targets.size() != 1 || targets[0] != layoutTarget) {
        module.emitError(
            "dispatch scheduling target differs from materialized layout "
            "target");
        return signalPassFailure();
      }
    }
    if (targets.size() != 1 || !isLLVMCPUBackend(targets[0])) {
      return;
    }
    if (explicitTarget && explicitTarget != targets[0]) {
      module.emitError(
          "dispatch scheduling target conflicts with device affinity");
      return signalPassFailure();
    }
    // The compute owner freezes this resolved configuration during the script.
    // Expose it only while scheduling; a host module is not an executable.
    module->setAttr("hal.executable.target", targets[0]);
    auto restoreTarget = llvm::scope_exit([&] {
      if (explicitTarget) {
        module->setAttr("hal.executable.target", explicitTarget);
      } else {
        module->removeAttr("hal.executable.target");
      }
    });
    auto environment = DictionaryAttr::get(
        &getContext(),
        {NamedAttribute(StringAttr::get(&getContext(), "library_sha256"),
                        StringAttr::get(&getContext(), sourceDigest)),
         NamedAttribute(StringAttr::get(&getContext(), "target"), targets[0])});
    // Identity survives outlining/cloning. Pointer identity alone would make
    // an existing, unscheduled payload look newly selected after outlining.
    DenseMap<Attribute, Attribute> previousPayloads;
    DenseMap<Attribute, DistinctAttr> preservationTokens;
    module.walk([&](PayloadRegionOp payload) {
      Attribute old = payload->getAttr(kPreserved);
      auto [it, inserted] = preservationTokens.try_emplace(old);
      if (inserted) {
        it->second = DistinctAttr::create(UnitAttr::get(&getContext()));
        previousPayloads[it->second] = old;
      }
      // Equal original attributes share a token so this bookkeeping does not
      // prevent otherwise equivalent, pre-existing bodies from deduplicating.
      payload->setAttr(kPreserved, it->second);
    });
    auto restoreTemporaryAttrs = llvm::scope_exit([&] {
      module.walk([&](PayloadRegionOp payload) {
        auto it = previousPayloads.find(payload->getAttr(kPreserved));
        if (it == previousPayloads.end()) {
          return;
        }
        if (it->second) {
          payload->setAttr(kPreserved, it->second);
        } else {
          payload->removeAttr(kPreserved);
        }
      });
    });
    if (failed(applySchedule(module, entry, library, environment))) {
      module.emitError(
          "selected dispatch scheduling schedule failed; classic "
          "fallback is not rollback");
      return signalPassFailure();
    }

    // Publish the immutable library before handing off to parallel workers or
    // serializing the selection checkpoint. No worker writes host attributes.
    bool hasSelection = false;
    module.walk([&](IREE::Flow::ExecutableExportOp op) {
      hasSelection |= op->hasAttr(kSchedule);
    });
    if (hasSelection) {
      NamedAttrList sources(module->getAttrOfType<DictionaryAttr>(kSources));
      sources.set(sourceDigest, StringAttr::get(&getContext(), librarySource));
      module->setAttr(kSources, sources.getDictionary(&getContext()));
    }
    if (stopAfterSelection) {
      return;
    }
    OpPassManager execution(ModuleOp::getOperationName());
    execution.addPass(createExecuteDispatchSchedulesPass());
    if (failed(runPipeline(execution, module))) {
      return signalPassFailure();
    }

    // Validate the scheduling cutoff even for user overrides. No later host
    // pass may silently finish numerical lowering on behalf of the dispatch
    // script.
    SmallVector<ModuleOp> computes;
    bool malformed = false;
    module.walk([&](PayloadRegionOp payload) {
      if (previousPayloads.contains(payload->getAttr(kPreserved))) {
        return;
      }
      auto owned = payload.getBody().front().getOps<ModuleOp>();
      if (!llvm::hasSingleElement(owned)) {
        payload.emitError(
            "dispatch schedule must produce one owned LLVM compute module");
        malformed = true;
        return;
      }
      computes.push_back(*owned.begin());
    });
    if (malformed) {
      return signalPassFailure();
    }
    for (ModuleOp compute : computes) {
      OpPassManager verifier(ModuleOp::getOperationName());
      verifier.addPass(createLLVMCPUVerifyPayloadCodegenPass());
      if (failed(runPipeline(verifier, compute))) {
        return signalPassFailure();
      }
      // Workers have already recorded their own library identity. Preserve it
      // when a module contains selections from several source snapshots.
      if (!compute->hasAttr("iree_payload.schedule_sha256")) {
        compute->setAttr("iree_payload.schedule_sha256",
                         StringAttr::get(&getContext(), sourceDigest));
      }
    }
    if (!computes.empty()) {
      NamedAttrList sources(module->getAttrOfType<DictionaryAttr>(
          "iree_payload.schedule_sources"));
      sources.set(sourceDigest, StringAttr::get(&getContext(), librarySource));
      module->setAttr("iree_payload.schedule_sources",
                      sources.getDictionary(&getContext()));
    }
  }

 private:
  ModuleOp library;
  transform::NamedSequenceOp entry;
  std::string sourceDigest;
};
}  // namespace
}  // namespace mlir::iree_compiler::Experimental
