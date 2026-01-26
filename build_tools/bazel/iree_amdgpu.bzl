# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AMDGPU SDK target compatibility helpers.

This module provides wrapper macros that automatically add target_compatible_with
constraints to ensure targets are skipped when AMDGPU SDK is not available.

Usage in BUILD files:
    load("//build_tools/bazel:iree_amdgpu.bzl", "iree_amdgpu_cc_library", "iree_amdgpu_cc_test")

    iree_amdgpu_cc_library(
        name = "my_amdgpu_lib",
        srcs = ["my_amdgpu_lib.c"],
        ...
    )

When AMDGPU is not available, targets will be skipped with:
    "Target //path:target is incompatible and will be skipped"
"""

load("//build_tools/bazel:build_defs.oss.bzl", "iree_runtime_cc_library", "iree_runtime_cc_test")
load("//build_tools/bazel:iree_amdgpu_binary.bzl", _iree_amdgpu_binary = "iree_amdgpu_binary")
load("//build_tools/embed_data:build_defs.bzl", "iree_c_embed_data")

def amdgpu_target_compatible_with():
    """Returns target_compatible_with for AMDGPU-dependent targets.

    When amdgpu_enabled is false, returns [@platforms//:incompatible] which
    makes the target incompatible with all platforms (i.e., it will be skipped).

    This function can be imported and used by other macros that need to
    conditionally skip AMDGPU-dependent targets.
    """
    return select({
        "//runtime/src/iree/hal/drivers:amdgpu_enabled": [],
        "//conditions:default": ["@platforms//:incompatible"],
    })

def iree_amdgpu_cc_library(target_compatible_with = [], **kwargs):
    """Wrapper for iree_runtime_cc_library that adds AMDGPU SDK compatibility.

    All arguments are passed through to iree_runtime_cc_library, with
    target_compatible_with automatically including the AMDGPU SDK constraint.

    Args:
        target_compatible_with: Additional target compatibility constraints.
        **kwargs: All other arguments passed to iree_runtime_cc_library.
    """
    iree_runtime_cc_library(
        target_compatible_with = target_compatible_with + amdgpu_target_compatible_with(),
        **kwargs
    )

def iree_amdgpu_cc_test(target_compatible_with = [], **kwargs):
    """Wrapper for iree_runtime_cc_test that adds AMDGPU SDK compatibility.

    All arguments are passed through to iree_runtime_cc_test, with
    target_compatible_with automatically including the AMDGPU SDK constraint.

    Args:
        target_compatible_with: Additional target compatibility constraints.
        **kwargs: All other arguments passed to iree_runtime_cc_test.
    """
    iree_runtime_cc_test(
        target_compatible_with = target_compatible_with + amdgpu_target_compatible_with(),
        **kwargs
    )

def iree_amdgpu_binary(target_compatible_with = [], **kwargs):
    """Wrapper for iree_amdgpu_binary that adds AMDGPU SDK compatibility.

    All arguments are passed through to the underlying iree_amdgpu_binary, with
    target_compatible_with automatically including the AMDGPU SDK constraint.

    Args:
        target_compatible_with: Additional target compatibility constraints.
        **kwargs: All other arguments passed to iree_amdgpu_binary.
    """
    _iree_amdgpu_binary(
        target_compatible_with = target_compatible_with + amdgpu_target_compatible_with(),
        **kwargs
    )

def iree_amdgpu_c_embed_data(target_compatible_with = [], **kwargs):
    """Wrapper for iree_c_embed_data that adds AMDGPU SDK compatibility.

    All arguments are passed through to iree_c_embed_data, with
    target_compatible_with automatically including the AMDGPU SDK constraint.

    Note: This requires iree_c_embed_data to support target_compatible_with,
    which it passes to the underlying cc_library. The genrule inside
    iree_c_embed_data will still be analyzed but won't execute if nothing
    depends on its outputs.

    Args:
        target_compatible_with: Additional target compatibility constraints.
        **kwargs: All other arguments passed to iree_c_embed_data.
    """
    iree_c_embed_data(
        target_compatible_with = target_compatible_with + amdgpu_target_compatible_with(),
        **kwargs
    )
