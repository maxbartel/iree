// RUN: iree-opt --iree-plugin=dispatch_scheduling %s --iree-convert-to-llvm --verify-diagnostics

func.func private @iree_uk_mmt4d(memref<f32>, index, index, memref<f32>, index, index, memref<f32>, index, index, index, index, index, i32, i32, i32, i32) -> i32 attributes {hal.import.bitcode = true, hal.import.fields = ["processor_data"], llvm.bareptr = true}
// expected-error @below {{conflicting declaration for payload bitcode import}}
llvm.func @iree_uk_mmt4d_0(!llvm.ptr, i64, i64, !llvm.ptr, i64, i64, !llvm.ptr, i64, i64, i64, i64, i64, i32, i32, i32, i32) -> i32 attributes {hal.import.bitcode = true, hal.import.fields = ["processor_id"], llvm.bareptr = true, iree_payload.import = "iree_uk_mmt4d"}
func.func @dispatch() {
  %ptr = llvm.mlir.zero : !llvm.ptr
  %i = llvm.mlir.constant(0 : i64) : i64
  %j = llvm.mlir.constant(0 : i32) : i32
  %r = llvm.call @iree_uk_mmt4d_0(%ptr, %i, %i, %ptr, %i, %i, %ptr, %i, %i, %i, %i, %i, %j, %j, %j, %j) : (!llvm.ptr, i64, i64, !llvm.ptr, i64, i64, !llvm.ptr, i64, i64, i64, i64, i64, i32, i32, i32, i32) -> i32
  return
}
