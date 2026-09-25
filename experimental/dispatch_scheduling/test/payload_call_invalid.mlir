// RUN: iree-opt --iree-plugin=dispatch_scheduling %s --split-input-file --verify-diagnostics

// missing ABI
module @compute {
  func.func @entry(%arg: memref<4xf32>) {
    return
  }
}
func.func @caller(%arg: memref<4xf32>) {
  // expected-error@+1 {{arguments must match the callee's void buffer ABI}}
  iree_payload.call @compute::@entry(%arg) : (memref<4xf32>) -> ()
  return
}

// -----

// mismatched operand type
module @compute {
  func.func @entry(%arg: memref<4xf32>) attributes {iree_payload.abi = (memref<4xf32>) -> ()} {
    return
  }
}
func.func @caller(%arg: memref<8xf32>) {
  // expected-error@+1 {{arguments must match the callee's void buffer ABI}}
  iree_payload.call @compute::@entry(%arg) : (memref<8xf32>) -> ()
  return
}

// -----

// nested private entry
module @compute {
  func.func private @entry(%arg: memref<4xf32>) attributes {iree_payload.abi = (memref<4xf32>) -> ()} {
    return
  }
}
func.func @caller(%arg: memref<4xf32>) {
  // expected-error@+1 {{callee must reference a defined compute function}}
  iree_payload.call @compute::@entry(%arg) : (memref<4xf32>) -> ()
  return
}

// -----

// mismatched unconverted function
module @compute {
  func.func @entry(%arg: memref<8xf32>) attributes {iree_payload.abi = (memref<4xf32>) -> ()} {
    return
  }
}
func.func @caller(%arg: memref<4xf32>) {
  // expected-error@+1 {{unconverted compute signature must match its buffer ABI}}
  iree_payload.call @compute::@entry(%arg) : (memref<4xf32>) -> ()
  return
}

// -----

// tensor ABI
module @compute {
  func.func @entry(%arg: tensor<4xf32>) attributes {iree_payload.abi = (tensor<4xf32>) -> ()} {
    return
  }
}
func.func @caller(%arg: tensor<4xf32>) {
  // expected-error@+1 {{compute ABI cannot contain tensor arguments}}
  iree_payload.call @compute::@entry(%arg) : (tensor<4xf32>) -> ()
  return
}
