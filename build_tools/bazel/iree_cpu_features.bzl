# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""CPU feature target compatibility helpers.

This module provides helper functions to conditionally skip tests that require
specific CPU features not available on the host.

Usage in BUILD files:
    load("//build_tools/bazel:iree_cpu_features.bzl", "arm_sme_test_target_compatible_with")

    iree_generated_e2e_runner_test(
        ...
        target_compatible_with = arm_sme_test_target_compatible_with(),
    )

When ARM SME runtime is not available, tests will be skipped with:
    "Target //path:target is incompatible and will be skipped"

To run ARM SME tests, set the runtime_available flag:
    bazel test //... --//runtime/src/iree/hal/local:arm_sme_runtime_available=True
"""

def arm_sme_test_target_compatible_with():
    """Returns target_compatible_with for ARM SME execution tests.

    When arm_sme_runtime_available is False (the default), returns [@platforms//:incompatible]
    which makes the target incompatible with all platforms (i.e., it will be skipped).
    """
    return select({
        "//runtime/src/iree/hal/local:arm_sme_runtime_available_setting": [],
        "//conditions:default": ["@platforms//:incompatible"],
    })
