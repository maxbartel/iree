// RUN: iree-opt %s --split-input-file --pass-pipeline="builtin.module(iree-llvmgpu-select-lowering-strategy)" \
// RUN:     --iree-codegen-llvmgpu-enable-transform-dialect-jit=1 --iree-codegen-llvmgpu-enable-transform-dialect-batch-matmul-strategy |\
// RUN:   FileCheck %s --check-prefixes=CHECK,DEFAULT

// RUN: iree-opt %s --split-input-file --pass-pipeline="builtin.module(iree-llvmgpu-select-lowering-strategy)" \
// RUN:     --iree-codegen-llvmgpu-enable-transform-dialect-jit=1 --iree-codegen-llvmgpu-enable-transform-dialect-batch-matmul-strategy \
// RUN: -td-matmul-strategy-blk-sizes=128,64,32,2 \
// RUN: -td-matmul-strategy-reduc-size=8 \
// RUN: -td-matmul-strategy-num-threads=32,4,1 \
// RUN: -td-matmul-strategy-num-warps=1,4,1 \
// RUN: -td-matmul-strategy-use-async-copies=true \
// RUN: -td-matmul-strategy-pipeline-depth=3 \
// RUN: -td-matmul-strategy-use-mma-sync=false \
// RUN: -td-matmul-strategy-use-fma=true \
// RUN:   | FileCheck %s --check-prefixes=CHECK,OPTIONS

#executable_target_cuda_nvptx_fb = #hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_80"}>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
module {
  func.func @batch_matmul_dispatch_0_generic_128x80x320x32_f32() attributes {hal.executable.target = #executable_target_cuda_nvptx_fb} {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<128x80x32xf32>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<128x32x320xf32>>
    %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<128x80x320xf32>>
    %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [128, 80, 32], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<128x80x32xf32>> -> tensor<128x80x32xf32>
    %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [128, 32, 320], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<128x32x320xf32>> -> tensor<128x32x320xf32>
    %5 = tensor.empty() : tensor<128x80x320xf32>
    %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<128x80x320xf32>) -> tensor<128x80x320xf32>
    %7 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%3, %4 : tensor<128x80x32xf32>, tensor<128x32x320xf32>) outs(%6 : tensor<128x80x320xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %8 = arith.mulf %in, %in_0 : f32
      %9 = arith.addf %out, %8 : f32
      linalg.yield %9 : f32
    } -> tensor<128x80x320xf32>
    flow.dispatch.tensor.store %7, %2, offsets = [0, 0, 0], sizes = [128, 80, 320], strides = [1, 1, 1] : tensor<128x80x320xf32> -> !flow.dispatch.tensor<writeonly:tensor<128x80x320xf32>>
    return
  }
}

// CHECK: transform.named_sequence
// CHECK:   transform.iree.register_match_callbacks
// CHECK:   %[[MATCH:.+]]:2 = transform.iree.match_callback failures(propagate) "batch_matmul"
// CHECK:   %[[TILED:.+]], %[[FORALL:.+]] = transform.structured.tile_using_forall %[[MATCH]]#1
// DEFAULT:   tile_sizes [64, 64, 1](mapping = [#gpu.block<z>, #gpu.block<y>, #gpu.block<x>])
// OPTIONS:   tile_sizes [128, 64, 32](mapping = [#gpu.block<z>, #gpu.block<y>, #gpu.block<x>])
// CHECK:   apply_patterns
// CHECK:   transform.iree.apply_licm
// CHECK:   transform.apply_cse to
// CHECK:   %[[FUSED:.+]], %[[CONTAINING:.+]] = transform.structured.fuse_into_containing_op %[[MATCH]]#0 into %[[FORALL]]
// CHECK:   transform.iree.populate_workgroup_count_region_using_num_threads_slice %[[FORALL]]
// CHECK:   %[[TILED_LINALG:.+]], %[[LOOPS:.+]] = transform.structured.tile_using_for %tiled_op
// DEFAULT:   [0, 0, 0, 16]
// OPTIONS:   [0, 0, 0, 8]
// CHECK:   %[[PADDED:.+]], %{{.*}}, %{{.+}} = transform.structured.pad %tiled_linalg_op
// CHECK:     pack_paddings = [1, 1, 1, 1], pad_to_multiple_of = [1, 1, 1, 1], padding_dimensions = [0, 1, 2, 3]
// CHECK:     padding_values = [0.000000e+00 : f32, 0.000000e+00 : f32, 0.000000e+00 : f32]}
// CHECK:   %[[V3:.+]] = transform.get_producer_of_operand %[[PADDED]][2]
// CHECK:   transform.structured.hoist_pad %{{.*}} by 1 loops
// CHECK:   apply_patterns
// CHECK:   transform.iree.apply_licm
// CHECK:   transform.apply_cse to
// CHECK:   %[[FILL:.+]] = transform.structured.match ops{["linalg.fill"]}
// CHECK:   apply_patterns
// CHECK:   transform.iree.apply_licm
// CHECK:   transform.apply_cse to
// CHECK:   transform.structured.match ops{["tensor.parallel_insert_slice"]}
// CHECK:   transform.structured.insert_slice_to_copy
// CHECK:   %[[LHS:.+]] = transform.get_producer_of_operand %[[PADDED]][0]
// CHECK:   %[[RHS:.+]] = transform.get_producer_of_operand %[[PADDED]][1]
// CHECK:   %[[RHS_DPS:.+]] = transform.structured.rewrite_in_destination_passing_style %[[RHS]]

// CHECK:   transform.structured.tile_using_forall %[[LHS]]
// DEFAULT:  num_threads [1, 32, 4](mapping = [#gpu.thread<linear_dim_2>, #gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>])
// OPTIONS:  num_threads [1, 64, 2](mapping = [#gpu.thread<linear_dim_2>, #gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>])
// CHECK:   apply_patterns
// CHECK:   transform.iree.apply_licm
// CHECK:   transform.apply_cse to
// CHECK:   transform.structured.match ops{["scf.if"]}
// CHECK:   transform.scf.take_assumed_branch %{{.*}} take_else_branch

// CHECK:   transform.structured.tile_using_forall %[[RHS_DPS]]
// DEFAULT:  num_threads [8, 16, 1](mapping = [#gpu.thread<linear_dim_2>, #gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>])
// OPTIONS:  num_threads [2, 8, 8](mapping = [#gpu.thread<linear_dim_2>, #gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>])
// CHECK:   apply_patterns
// CHECK:   transform.iree.apply_licm
// CHECK:   transform.apply_cse to

// CHECK:   transform.structured.tile_using_forall
// DEFAULT:  num_threads [2, 64, 1](mapping = [#gpu.thread<linear_dim_2>, #gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>])
// OPTIONS:  num_threads [1, 16, 8](mapping = [#gpu.thread<linear_dim_2>, #gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>])
// CHECK:   apply_patterns
// CHECK:   transform.iree.apply_licm
// CHECK:   transform.apply_cse to

// CHECK:   transform.structured.tile_using_forall
// DEFAULT:  num_threads [1, 2, 64](mapping = [#gpu.thread<z>, #gpu.thread<y>, #gpu.thread<x>])
// OPTIONS:  num_threads [1, 4, 32](mapping = [#gpu.thread<z>, #gpu.thread<y>, #gpu.thread<x>])
// CHECK:   apply_patterns
// CHECK:   transform.iree.apply_licm
// CHECK:   transform.apply_cse to

// CHECK:   %tiled_op_8, %forall_op_9 = transform.structured.tile_using_forall %[[FILL]]
// DEFAULT:   num_threads [1, 2, 64](mapping = [#gpu.thread<z>, #gpu.thread<y>, #gpu.thread<x>])
// OPTIONS:   num_threads [1, 4, 32](mapping = [#gpu.thread<z>, #gpu.thread<y>, #gpu.thread<x>])
// CHECK:   apply_patterns
// CHECK:   transform.iree.apply_licm
// CHECK:   transform.apply_cse to
// CHECK:   apply_patterns
// CHECK:   transform.iree.apply_licm
// CHECK:   transform.apply_cse to

// CHECK:   transform.structured.vectorize
// DEFAULT:   vector_sizes [64, 2, 4]
// OPTIONS:   vector_sizes [128, 1, 4]
// CHECK:   transform.structured.vectorize
// DEFAULT:   vector_sizes [32, 1, 1]
// OPTIONS:   vector_sizes [128, 4, 4]
// CHECK:   apply_patterns
// CHECK:     transform.apply_patterns.vector.lower_masked_transfers
// CHECK:   transform.structured.vectorize_children_and_apply_patterns
// CHECK:   apply_patterns
// CHECK:   transform.iree.apply_licm
// CHECK:   transform.apply_cse to
// CHECK:   apply_patterns
// CHECK:     transform.apply_patterns.canonicalization
// CHECK:   transform.iree.apply_licm
// CHECK:   transform.apply_cse to
// CHECK:   transform.iree.eliminate_empty_tensors

// CHECK:   transform.iree.bufferize {target_gpu}
// CHECK:   transform.memref.erase_dead_alloc_and_stores
// CHECK:   transform.iree.forall_to_workgroup
// CHECK:   transform.iree.map_nested_forall_to_gpu_threads
// DEFAULT:  workgroup_dims = [64, 2, 1]
// OPTIONS:  workgroup_dims = [32, 4, 1]
// CHECK:   transform.iree.eliminate_gpu_barriers
// CHECK:   apply_patterns
// CHECK:   transform.iree.apply_licm
// CHECK:   transform.apply_cse to
// CHECK:   transform.iree.hoist_static_alloc
// CHECK:   apply_patterns
// CHECK:     transform.apply_patterns.memref.fold_memref_alias_ops
// CHECK:   apply_patterns
// CHECK:     transform.apply_patterns.memref.extract_address_computations
// CHECK:   apply_patterns
// CHECK:     transform.apply_patterns.linalg.tiling_canonicalization
// CHECK:     transform.apply_patterns.iree.fold_fill_into_pad
// CHECK:     transform.apply_patterns.scf.for_loop_canonicalization
// CHECK:     transform.apply_patterns.canonicalization
// CHECK:   transform.iree.apply_licm
// CHECK:   transform.apply_cse to
// CHECK:   transform.iree.synchronize_loop
// CHECK:   transform.structured.hoist_redundant_vector_transfers
// CHECK:   apply_patterns
// CHECK:   transform.iree.apply_licm
// CHECK:   transform.apply_cse to
// CHECK:   transform.memref.erase_dead_alloc_and_stores
// CHECK:   transform.iree.eliminate_gpu_barriers
// CHECK:   apply_patterns
// CHECK:   transform.iree.apply_licm
// CHECK:   transform.apply_cse to
// CHECK:   apply_patterns
// CHECK:     transform.apply_patterns.memref.fold_memref_alias_ops

// CHECK:   transform.memref.multibuffer
// DEFAULT:   factor = 2
// OPTIONS:   factor = 3
// CHECK:   apply_patterns
// CHECK:     transform.apply_patterns.vector.transfer_to_scf   full_unroll = true
// CHECK:   apply_patterns
// CHECK:   transform.iree.apply_licm
// CHECK:   transform.apply_cse to
// CHECK:   transform.iree.create_async_groups
// CHECK:   apply_patterns
// CHECK:   transform.iree.apply_licm
// CHECK:   transform.apply_cse to
// CHECK:   transform.iree.pipeline_shared_memory_copies
// DEFAULT:   depth = 2
// OPTIONS:   depth = 3
// CHECK:   apply_patterns
// CHECK:     transform.apply_patterns.vector.lower_masks
// CHECK:   apply_patterns
// CHECK:     transform.apply_patterns.vector.materialize_masks
// CHECK:   apply_patterns
// CHECK:   transform.iree.apply_licm
// CHECK:   transform.apply_cse to
