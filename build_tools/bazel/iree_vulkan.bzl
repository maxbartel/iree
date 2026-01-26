# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Vulkan target compatibility helpers.

This module provides helper functions to conditionally skip Vulkan execution
tests when Vulkan runtime is not available.

Usage in BUILD files:
    load("//build_tools/bazel:iree_vulkan.bzl", "vulkan_test_target_compatible_with")

    iree_check_single_backend_test_suite(
        ...
        target_compatible_with = vulkan_test_target_compatible_with(),
    )

When Vulkan runtime is not available, tests will be skipped with:
    "Target //path:target is incompatible and will be skipped"

To run Vulkan execution tests, set the runtime_available flag:
    bazel test //... --//runtime/src/iree/hal/drivers/vulkan:runtime_available=True
"""

def vulkan_test_target_compatible_with():
    """Returns target_compatible_with for Vulkan execution tests.

    When runtime_available is False (the default), returns [@platforms//:incompatible]
    which makes the target incompatible with all platforms (i.e., it will be skipped).

    Vulkan libraries still build normally - this only affects test targets.
    """
    return select({
        "//runtime/src/iree/hal/drivers/vulkan:runtime_available_setting": [],
        "//conditions:default": ["@platforms//:incompatible"],
    })
