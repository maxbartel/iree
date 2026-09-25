// RUN: iree-opt --iree-plugin=dispatch_scheduling %s --transform-interpreter | FileCheck %s --implicit-check-not=memref.copy --implicit-check-not=memref.alloc

// CFG conversion occurs only in the owned function; the DPS boundary and exact
// view types survive. The compute function receives dynamic sizes explicitly.
// CHECK-LABEL: func.func @owned(
// CHECK: iree_payload.region
// CHECK-NEXT: ^bb0(%[[IN:.*]]: memref<?xf32, strided<[2], offset: ?>>, %[[N:.*]]: index, %[[OUT:.*]]: memref<?xf32, strided<[2], offset: ?>>):
// CHECK-NEXT: module @compute attributes {iree_payload.compute}
// CHECK: func.func @entry(
// CHECK-SAME: iree_payload.abi = (memref<?xf32, strided<[2], offset: ?>>, index, memref<?xf32, strided<[2], offset: ?>>) -> ()
// CHECK-SAME: iree_payload.num_inputs = 2 : i64
// CHECK: cf.br
// CHECK: cf.cond_br
// CHECK: memref.load
// CHECK-NEXT: memref.store
// CHECK: iree_payload.call @compute::@entry(%[[IN]], %[[N]], %[[OUT]]) : (memref<?xf32, strided<[2], offset: ?>>, index, memref<?xf32, strided<[2], offset: ?>>) -> ()
// CHECK-NEXT: iree_payload.yield
func.func @owned(%input: memref<?xf32, strided<[2], offset: ?>>, %output: memref<?xf32, strided<[2], offset: ?>>, %n: index) {
  iree_payload.region ins(%input, %n : memref<?xf32, strided<[2], offset: ?>>, index) outs(%output : memref<?xf32, strided<[2], offset: ?>>) {
  ^bb0(%a: memref<?xf32, strided<[2], offset: ?>>, %size: index, %b: memref<?xf32, strided<[2], offset: ?>>):
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %size step %c1 {
      %v = memref.load %a[%i] : memref<?xf32, strided<[2], offset: ?>>
      memref.store %v, %b[%i] : memref<?xf32, strided<[2], offset: ?>>
    }
    iree_payload.yield
  }
  return
}
// Multiple payloads own independent @compute modules. A pre-existing @entry
// helper keeps its symbol; the new designated entry gets a collision-free name.
// CHECK-LABEL: func.func @symbols()
// CHECK: module @compute
// CHECK: func.func private @entry()
// CHECK: func.func @entry_0()
// CHECK: call @entry()
// CHECK: iree_payload.call @compute::@entry_0()
func.func @symbols() {
  iree_payload.region {
    func.func private @entry() {
      return
    }
    func.call @entry() : () -> ()
    iree_payload.yield
  }
  return
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %p = transform.structured.match ops{["iree_payload.region"]} in %root : (!transform.any_op) -> !transform.any_op
    %m = transform.iree.outline_payload_region %p : (!transform.any_op) -> !transform.any_op
    %cf = transform.apply_registered_pass "convert-scf-to-cf" to %m : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
