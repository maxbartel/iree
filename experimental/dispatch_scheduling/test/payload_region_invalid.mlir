// RUN: iree-opt --iree-plugin=dispatch_scheduling %s --split-input-file --verify-diagnostics

func.func @tensor_result_without_tied_out(%arg0: tensor<4xf32>, %init0: tensor<4xf32>)
    -> (tensor<4xf32>, tensor<4xf32>) {
  // expected-error@+1 {{result count must match tensor output count}}
  %0, %1 = iree_payload.region
      ins(%arg0 : tensor<4xf32>)
      outs(%init0 : tensor<4xf32>) {
  ^bb0(%arg0_b: tensor<4xf32>, %init0_b: tensor<4xf32>):
    iree_payload.yield %init0_b, %init0_b : tensor<4xf32>, tensor<4xf32>
  } -> tensor<4xf32>, tensor<4xf32>
  return %0, %1 : tensor<4xf32>, tensor<4xf32>
}

// -----

func.func @memref_result(%arg0: memref<4xf32>, %out0: memref<4xf32>) -> tensor<4xf32> {
  // expected-error@+1 {{result count must match tensor output count}}
  %0 = iree_payload.region
      ins(%arg0 : memref<4xf32>)
      outs(%out0 : memref<4xf32>) {
  ^bb0(%arg0_b: memref<4xf32>, %out0_b: memref<4xf32>):
    %empty = tensor.empty() : tensor<4xf32>
    iree_payload.yield %empty : tensor<4xf32>
  } -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

func.func @tensor_capture_from_above(%arg0: tensor<4xf32>, %init0: tensor<4xf32>,
                                     %outer: tensor<4xf32>) -> tensor<4xf32> {
  // expected-note@+1 {{required by region isolation constraints}}
  %0 = iree_payload.region
      ins(%arg0 : tensor<4xf32>)
      outs(%init0 : tensor<4xf32>) {
  ^bb0(%arg0_b: tensor<4xf32>, %init0_b: tensor<4xf32>):
    // expected-error@+1 {{using value defined outside the region}}
    iree_payload.yield %outer : tensor<4xf32>
  } -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

func.func @memref_capture_from_above(%arg0: memref<4xf32>, %out0: memref<4xf32>,
                                     %outer: memref<4xf32>) {
  // expected-note@+1 {{required by region isolation constraints}}
  iree_payload.region
      ins(%arg0 : memref<4xf32>)
      outs(%out0 : memref<4xf32>) {
  ^bb0(%arg0_b: memref<4xf32>, %out0_b: memref<4xf32>):
    // expected-error@+1 {{using value defined outside the region}}
    memref.copy %outer, %out0_b : memref<4xf32> to memref<4xf32>
    iree_payload.yield
  }
  return
}

// -----

func.func @block_arg_count_too_few(%arg0: tensor<4xf32>, %init0: tensor<4xf32>)
    -> tensor<4xf32> {
  // expected-error@+1 {{entry block argument count must match ins and outs}}
  %0 = iree_payload.region
      ins(%arg0 : tensor<4xf32>)
      outs(%init0 : tensor<4xf32>) {
  ^bb0(%init0_b: tensor<4xf32>):
    iree_payload.yield %init0_b : tensor<4xf32>
  } -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

func.func @block_arg_type_mismatch(%arg0: tensor<4xf32>, %init0: tensor<4xf32>)
    -> tensor<4xf32> {
  // expected-error@+1 {{entry block argument types must match ins and outs}}
  %0 = iree_payload.region
      ins(%arg0 : tensor<4xf32>)
      outs(%init0 : tensor<4xf32>) {
  ^bb0(%arg0_b: tensor<8xf32>, %init0_b: tensor<4xf32>):
    iree_payload.yield %init0_b : tensor<4xf32>
  } -> tensor<4xf32>
  return %0 : tensor<4xf32>
}
