// RUN: iree-opt --iree-plugin=dispatch_scheduling %s --split-input-file --transform-interpreter | FileCheck %s

// The internal integer temporary is owned by the payload. Only its dynamic
// extent crosses the interface; the final destination still comes from outside.
// CHECK-LABEL: func.func @internal_temporary(
// CHECK-SAME: %[[DST:[^ ,:]+]]: tensor<?xf32>, %[[N:[^ ,:]+]]: index)
// CHECK: iree_payload.region ins(%[[N]] : index) outs(%[[DST]] : tensor<?xf32>)
// CHECK-NEXT: ^bb0(%[[SIZE:[^ ,:]+]]: index, %[[OUT:[^ ,:]+]]: tensor<?xf32>):
// CHECK: %[[EMPTY:.*]] = tensor.empty(%[[SIZE]]) : tensor<?xi32>
// CHECK: linalg.fill ins(%{{.*}} : i32) outs(%[[EMPTY]] : tensor<?xi32>)
// CHECK: linalg.generic
// CHECK-SAME: outs(%[[OUT]] : tensor<?xf32>)
// CHECK: iree_payload.yield
func.func @internal_temporary(%dst: tensor<?xf32>, %n: index) -> tensor<?xf32> {
  %empty = tensor.empty(%n) : tensor<?xi32>
  %one = arith.constant 1 : i32
  %filled = linalg.fill ins(%one : i32) outs(%empty : tensor<?xi32>) -> tensor<?xi32>
  %result = linalg.generic {indexing_maps = [affine_map<(i)->(i)>, affine_map<(i)->(i)>], iterator_types = ["parallel"]}
      ins(%filled : tensor<?xi32>) outs(%dst : tensor<?xf32>) {
  ^bb0(%v: i32, %out: f32):
    %f = arith.sitofp %v : i32 to f32
    linalg.yield %f : f32
  } -> tensor<?xf32>
  return %result : tensor<?xf32>
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %ops = transform.structured.match ops{["linalg.fill", "linalg.generic"]} in %root : (!transform.any_op) -> !transform.any_op
    %payload = transform.iree.wrap_in_payload_group %ops : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// Keep reads of the old destination distinct from writes through its DPS chain.
// Also exercise an empty selected group.
// CHECK-LABEL: func.func @chain_and_old_input(
// CHECK-SAME: %[[OUT:.*]]: tensor<4xf32>, %[[FACTOR:.*]]: f32)
// CHECK: %[[R:.*]] = iree_payload.region ins(%[[FACTOR]], %[[OUT]] : f32, tensor<4xf32>) outs(%[[OUT]] : tensor<4xf32>)
// CHECK-NEXT: ^bb0(%[[F:.*]]: f32, %[[OLD:.*]]: tensor<4xf32>, %[[DST:.*]]: tensor<4xf32>):
// CHECK-NEXT: %[[FILLED:.*]] = linalg.fill ins(%[[F]] : f32) outs(%[[DST]] : tensor<4xf32>)
// CHECK-NEXT: %[[SUM:.*]] = linalg.generic
// CHECK-SAME: ins(%[[OLD]], %[[FILLED]] : tensor<4xf32>, tensor<4xf32>) outs(%[[FILLED]] : tensor<4xf32>)
// CHECK: iree_payload.yield %[[SUM]] : tensor<4xf32>
// CHECK-NEXT: } -> tensor<4xf32>
// CHECK-NEXT: return %[[R]]
func.func @chain_and_old_input(%out: tensor<4xf32>, %factor: f32) -> tensor<4xf32> {
  %fill = linalg.fill ins(%factor : f32) outs(%out : tensor<4xf32>) -> tensor<4xf32>
  %r = linalg.generic {indexing_maps = [affine_map<(i)->(i)>, affine_map<(i)->(i)>, affine_map<(i)->(i)>], iterator_types = ["parallel"]} ins(%out, %fill : tensor<4xf32>, tensor<4xf32>) outs(%fill : tensor<4xf32>) {
  ^bb0(%old: f32, %value: f32, %unused: f32):
    %sum = arith.addf %old, %value : f32
    linalg.yield %sum : f32
  } -> tensor<4xf32>
  return %r : tensor<4xf32>
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %ops = transform.structured.match ops{["linalg.fill", "linalg.generic"]} in %root : (!transform.any_op) -> !transform.any_op
    %payload = transform.iree.wrap_in_payload_group %ops : (!transform.any_op) -> !transform.any_op
    %none = transform.structured.match ops{["test.absent"]} in %root : (!transform.any_op) -> !transform.any_op
    %empty = transform.iree.wrap_in_payload_group %none : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
// -----
// Keep tensor weights outside the isolated body; rematerialize scalar constants
// used by the selected operations or captured inside their regions.
// CHECK-LABEL: func.func @external_tensor_constant(
// CHECK-SAME: %[[OUT:.*]]: tensor<4xf32>)
// CHECK: %[[WEIGHTS:.*]] = arith.constant dense<2.000000e+00> : tensor<4xf32>
// CHECK: %[[R:.*]] = iree_payload.region ins(%[[WEIGHTS]] : tensor<4xf32>) outs(%[[OUT]] : tensor<4xf32>)
// CHECK-NEXT: ^bb0(%[[W:.*]]: tensor<4xf32>, %[[DST:.*]]: tensor<4xf32>):
// CHECK-NEXT: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT: %[[ONE:.*]] = arith.constant 1.000000e+00 : f32
// CHECK-NEXT: %[[FILLED:.*]] = linalg.fill ins(%[[ZERO]] : f32) outs(%[[DST]] : tensor<4xf32>)
// CHECK-NEXT: %[[SUM:.*]] = linalg.generic
// CHECK-SAME: ins(%[[W]] : tensor<4xf32>) outs(%[[FILLED]] : tensor<4xf32>)
// CHECK: arith.addf %{{.*}}, %[[ONE]] : f32
// CHECK: iree_payload.yield %[[SUM]] : tensor<4xf32>
// CHECK-NEXT: } -> tensor<4xf32>
// CHECK-NEXT: return %[[R]]
func.func @external_tensor_constant(%out: tensor<4xf32>) -> tensor<4xf32> {
  %weights = arith.constant dense<2.0> : tensor<4xf32>
  %zero = arith.constant 0.0 : f32
  %one = arith.constant 1.0 : f32
  %fill = linalg.fill ins(%zero : f32) outs(%out : tensor<4xf32>) -> tensor<4xf32>
  %r = linalg.generic {indexing_maps = [affine_map<(i)->(i)>, affine_map<(i)->(i)>], iterator_types = ["parallel"]} ins(%weights : tensor<4xf32>) outs(%fill : tensor<4xf32>) {
  ^bb0(%w: f32, %v: f32):
    %sum = arith.addf %w, %v : f32
    %result = arith.addf %sum, %one : f32
    linalg.yield %result : f32
  } -> tensor<4xf32>
  return %r : tensor<4xf32>
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %epilogue = transform.structured.match ops{["linalg.generic"]} in %root : (!transform.any_op) -> !transform.any_op
    %fill = transform.structured.match ops{["linalg.fill"]} in %root : (!transform.any_op) -> !transform.any_op
    %ops = transform.merge_handles %epilogue, %fill : !transform.any_op
    %payload = transform.iree.wrap_in_payload_group %ops : (!transform.any_op) -> !transform.any_op
    %none = transform.structured.match ops{["test.absent"]} in %root : (!transform.any_op) -> !transform.any_op
    %empty = transform.iree.wrap_in_payload_group %none : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// Multiple live results retain separate DPS destinations. The FP32 maximum
// remains available to quantization while its FP16 scale is also returned.
// CHECK-LABEL: func.func @max_and_scale(
// CHECK-SAME: %[[X:[^ ,:]+]]: tensor<2x32xf32>, %[[MAX_OUT:[^ ,:]+]]: tensor<2xf32>, %[[SCALE_OUT:[^ ,:]+]]: tensor<2xf16>)
// CHECK: %[[R:.*]]:2 = iree_payload.region ins(%[[X]] : tensor<2x32xf32>) outs(%[[MAX_OUT]], %[[SCALE_OUT]] : tensor<2xf32>, tensor<2xf16>)
// CHECK: %[[MAX:.*]] = linalg.generic
// CHECK: %[[SCALE:.*]] = linalg.generic
// CHECK-SAME: ins(%[[MAX]] : tensor<2xf32>)
// CHECK: iree_payload.yield %[[MAX]], %[[SCALE]] : tensor<2xf32>, tensor<2xf16>
// CHECK: return %[[R]]#0, %[[R]]#1
func.func @max_and_scale(%x: tensor<2x32xf32>, %max_out: tensor<2xf32>, %scale_out: tensor<2xf16>) -> (tensor<2xf32>, tensor<2xf16>) {
  %zero = arith.constant 0.0 : f32
  %init = linalg.fill ins(%zero : f32) outs(%max_out : tensor<2xf32>) -> tensor<2xf32>
  %max = linalg.generic {indexing_maps = [affine_map<(g,k)->(g,k)>, affine_map<(g,k)->(g)>], iterator_types = ["parallel", "reduction"]} ins(%x : tensor<2x32xf32>) outs(%init : tensor<2xf32>) {
  ^bb0(%v: f32, %a: f32):
    %abs = math.absf %v : f32
    %maxv = arith.maximumf %abs, %a : f32
    linalg.yield %maxv : f32
  } -> tensor<2xf32>
  %scale = linalg.generic {indexing_maps = [affine_map<(g)->(g)>, affine_map<(g)->(g)>], iterator_types = ["parallel"]} ins(%max : tensor<2xf32>) outs(%scale_out : tensor<2xf16>) {
  ^bb0(%m: f32, %unused: f16):
    %range = arith.constant 127.0 : f32
    %s = arith.divf %m, %range : f32
    %h = arith.truncf %s : f32 to f16
    linalg.yield %h : f16
  } -> tensor<2xf16>
  return %max, %scale : tensor<2xf32>, tensor<2xf16>
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %ops = transform.structured.match ops{["linalg.fill", "linalg.generic"]} in %root : (!transform.any_op) -> !transform.any_op
    %payload = transform.iree.wrap_in_payload_group %ops : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
