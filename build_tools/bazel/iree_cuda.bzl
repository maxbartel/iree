# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""CUDA SDK target compatibility helpers.

This module provides wrapper macros that automatically add target_compatible_with
constraints to ensure targets are skipped when CUDA SDK is not available.

Usage in BUILD files:
    load("//build_tools/bazel:iree_cuda.bzl", "iree_cuda_cc_library", "iree_cuda_cc_test")

    iree_cuda_cc_library(
        name = "my_cuda_lib",
        srcs = ["my_cuda_lib.c"],
        ...
    )

When CUDA is not available, targets will be skipped with:
    "Target //path:target is incompatible and will be skipped"
"""

load("//build_tools/bazel:build_defs.oss.bzl", "iree_runtime_cc_library", "iree_runtime_cc_test")

def cuda_target_compatible_with():
    """Returns target_compatible_with for CUDA-dependent targets.

    When cuda_enabled is false, returns [@platforms//:incompatible] which
    makes the target incompatible with all platforms (i.e., it will be skipped).

    This function can be imported and used by other macros that need to
    conditionally skip CUDA-dependent targets.
    """
    return select({
        "//runtime/src/iree/hal/drivers:cuda_enabled": [],
        "//conditions:default": ["@platforms//:incompatible"],
    })

def iree_cuda_cc_library(target_compatible_with = [], **kwargs):
    """Wrapper for iree_runtime_cc_library that adds CUDA SDK compatibility.

    All arguments are passed through to iree_runtime_cc_library, with
    target_compatible_with automatically including the CUDA SDK constraint.

    Args:
        target_compatible_with: Additional target compatibility constraints.
        **kwargs: All other arguments passed to iree_runtime_cc_library.
    """
    iree_runtime_cc_library(
        target_compatible_with = target_compatible_with + cuda_target_compatible_with(),
        **kwargs
    )

def iree_cuda_cc_test(target_compatible_with = [], **kwargs):
    """Wrapper for iree_runtime_cc_test that adds CUDA SDK compatibility.

    All arguments are passed through to iree_runtime_cc_test, with
    target_compatible_with automatically including the CUDA SDK constraint.

    Args:
        target_compatible_with: Additional target compatibility constraints.
        **kwargs: All other arguments passed to iree_runtime_cc_test.
    """
    iree_runtime_cc_test(
        target_compatible_with = target_compatible_with + cuda_target_compatible_with(),
        **kwargs
    )
