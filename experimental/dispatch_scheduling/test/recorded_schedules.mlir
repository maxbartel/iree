// RUN: iree-opt --iree-plugin=dispatch_scheduling %s --iree-dispatch-scheduling-select='library-file=%S/Inputs/schedule_choices.mlir stop-after-selection' -o %t.selected
// RUN: iree-opt --iree-plugin=dispatch_scheduling %t.selected --iree-dispatch-scheduling-execute -o %t.parallel
// RUN: iree-opt --iree-plugin=dispatch_scheduling %t.selected --iree-dispatch-scheduling-execute --mlir-disable-threading -o %t.serial
// RUN: diff %t.parallel %t.serial

// RUN: FileCheck %s --input-file=%t.parallel --implicit-check-not="flow.executable private @duplicate" --implicit-check-not=iree_payload.preserved
// RUN: iree-opt --iree-plugin=dispatch_scheduling %t.parallel --iree-dispatch-scheduling-execute -o %t.repeated
// RUN: diff %t.parallel %t.repeated
// RUN: sed '/^    } attributes/s/library_sha256 = "/library_sha256 = "0/g' %t.selected > %t.missing
// RUN: not iree-opt --iree-plugin=dispatch_scheduling %t.missing --iree-dispatch-scheduling-execute 2>&1 | FileCheck %s --check-prefix=MISSING
// RUN: sed '/^    } attributes/s/entry_point = "schedule_b"/entry_point = "absent"/' %t.selected > %t.entry
// RUN: not iree-opt --iree-plugin=dispatch_scheduling %t.entry --iree-dispatch-scheduling-execute 2>&1 | FileCheck %s --check-prefix=ENTRY
// RUN: sed '/^    } attributes/s/configuration = {tile = 2 : i64}/configuration = 2 : i64/' %t.selected > %t.config
// RUN: not iree-opt --iree-plugin=dispatch_scheduling %t.config --iree-dispatch-scheduling-execute --mlir-print-ir-after-failure 2>&1 | FileCheck %s --check-prefix=CONFIG --implicit-check-not="} {test.executed_"
// RUN: sed '/^    } attributes/s/target = #executable_target_embedded_elf_arm_64/target = #hal.executable.target<"llvm-cpu", "other">/g' %t.selected > %t.target
// RUN: not iree-opt --iree-plugin=dispatch_scheduling %t.target --iree-dispatch-scheduling-execute 2>&1 | FileCheck %s --check-prefix=TARGET
// RUN: sed '/^    } attributes/s/entry_point = "schedule_b"/entry_point = "schedule_a"/' %t.parallel > %t.changed
// RUN: not iree-opt --iree-plugin=dispatch_scheduling %t.changed --iree-dispatch-scheduling-execute 2>&1 | FileCheck %s --check-prefix=CHANGED

// CHECK-LABEL: flow.executable private @first
// CHECK: iree_payload.schedule_completed
// CHECK: } {test.executed_a = {tile = 1 : i64}}
// CHECK-LABEL: flow.executable private @different_entry
// CHECK: iree_payload.schedule_completed
// CHECK: } {test.executed_b = {tile = 1 : i64}}
// CHECK-LABEL: flow.executable private @different_config
// CHECK: iree_payload.schedule_completed
// CHECK: } {test.executed_a = {tile = 2 : i64}}
// MISSING: missing or mismatched dispatch schedule source snapshot
// ENTRY: missing recorded dispatch schedule entry: absent
// CONFIG: malformed dispatch schedule selection or environment
// TARGET: recorded dispatch schedule target differs from resolved target
// CHANGED: completed dispatch schedule selection changed

// This fixture tests schedule routing and equivalence, independently of
// numerical lowering (covered by the CPU schedule tests). Its owned
// body is already LLVM. Entry point and configuration differences must prevent
// deduplication; identical selections must share a body.
#target = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {
  target_triple = "aarch64-unknown-unknown-eabi", data_layout = "e-m:e-i64:64-i128:128-n32:64-S128", cpu = "generic", cpu_features = "+neon"
}>
module attributes {hal.executable.target = #target} {
  flow.executable private @first {
    flow.executable.export public @first workgroups() -> (index, index, index) {
      %c1 = arith.constant 1 : index
      flow.return %c1, %c1, %c1 : index, index, index
    }
    builtin.module {
      func.func @first() {
        iree_payload.region {
          builtin.module @compute attributes {hal.executable.target = #target,
              iree_payload.compute, iree_payload.prepared,
              iree_payload.target = #target} {
            llvm.func @entry() attributes {iree_payload.abi = () -> (), iree_payload.num_inputs = 0 : i64} {
              llvm.return
            }
          }
          iree_payload.call @compute::@entry() : () -> ()
          iree_payload.yield
        }
        return
      }
    }
  }
  flow.executable private @different_entry {
    flow.executable.export public @different_entry workgroups() -> (index, index, index) {
      %c1 = arith.constant 1 : index
      flow.return %c1, %c1, %c1 : index, index, index
    }
    builtin.module {
      func.func @different_entry() {
        iree_payload.region {
          builtin.module @compute attributes {hal.executable.target = #target,
              iree_payload.compute, iree_payload.prepared,
              iree_payload.target = #target} {
            llvm.func @entry() attributes {iree_payload.abi = () -> (), iree_payload.num_inputs = 0 : i64} {
              llvm.return
            }
          }
          iree_payload.call @compute::@entry() : () -> ()
          iree_payload.yield
        }
        return
      }
    }
  }
  flow.executable private @different_config {
    flow.executable.export public @different_config workgroups() -> (index, index, index) {
      %c1 = arith.constant 1 : index
      flow.return %c1, %c1, %c1 : index, index, index
    }
    builtin.module {
      func.func @different_config() {
        iree_payload.region {
          builtin.module @compute attributes {hal.executable.target = #target,
              iree_payload.compute, iree_payload.prepared,
              iree_payload.target = #target} {
            llvm.func @entry() attributes {iree_payload.abi = () -> (), iree_payload.num_inputs = 0 : i64} {
              llvm.return
            }
          }
          iree_payload.call @compute::@entry() : () -> ()
          iree_payload.yield
        }
        return
      }
    }
  }
  flow.executable private @duplicate {
    flow.executable.export public @duplicate workgroups() -> (index, index, index) {
      %c1 = arith.constant 1 : index
      flow.return %c1, %c1, %c1 : index, index, index
    }
    builtin.module {
      func.func @duplicate() {
        iree_payload.region {
          builtin.module @compute attributes {hal.executable.target = #target,
              iree_payload.compute, iree_payload.prepared,
              iree_payload.target = #target} {
            llvm.func @entry() attributes {iree_payload.abi = () -> (), iree_payload.num_inputs = 0 : i64} {
              llvm.return
            }
          }
          iree_payload.call @compute::@entry() : () -> ()
          iree_payload.yield
        }
        return
      }
    }
  }
}
