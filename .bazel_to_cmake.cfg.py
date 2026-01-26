# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import bazel_to_cmake_converter
import bazel_to_cmake_targets

DEFAULT_ROOT_DIRS = ["compiler", "runtime", "samples", "tests", "tools"]

REPO_MAP = {
    # Since this is the @iree_core repo, map to empty since all internal
    # targets are of the form "//compiler", not "@iree_core//compiler".
    "@iree_core": "",
}


class CustomBuildFileFunctions(bazel_to_cmake_converter.BuildFileFunctions):
    # target_compatible_with is Bazel-specific; ignore in CMake conversion.
    def _strip_target_compatible_with(self, kwargs):
        kwargs.pop("target_compatible_with", None)
        return kwargs

    # These functions return Bazel target_compatible_with constraints.
    # They are no-ops for CMake conversion.
    def cuda_target_compatible_with(self):
        return []

    def vulkan_test_target_compatible_with(self):
        return []

    def amdgpu_target_compatible_with(self):
        return []

    def arm_sme_test_target_compatible_with(self):
        return []

    # CUDA wrappers - pass through to underlying implementation
    def iree_cuda_cc_library(self, **kwargs):
        self.iree_runtime_cc_library(**self._strip_target_compatible_with(kwargs))

    def iree_cuda_cc_test(self, **kwargs):
        self.iree_runtime_cc_test(**self._strip_target_compatible_with(kwargs))

    # AMDGPU wrappers - pass through to underlying implementation
    def iree_amdgpu_cc_library(self, **kwargs):
        self.iree_runtime_cc_library(**self._strip_target_compatible_with(kwargs))

    def iree_amdgpu_cc_test(self, **kwargs):
        self.iree_runtime_cc_test(**self._strip_target_compatible_with(kwargs))

    def iree_amdgpu_binary(self, **kwargs):
        super().iree_amdgpu_binary(**self._strip_target_compatible_with(kwargs))

    def iree_amdgpu_c_embed_data(self, **kwargs):
        self.iree_c_embed_data(**self._strip_target_compatible_with(kwargs))

    def native_test(self, **kwargs):
        super().native_test(**self._strip_target_compatible_with(kwargs))

    def iree_compiler_cc_library(self, deps=[], **kwargs):
        self.cc_library(deps=deps + ["//compiler/src:defs"], **kwargs)

    def iree_runtime_cc_library(self, deps=[], **kwargs):
        self.cc_library(deps=deps + ["//runtime/src:runtime_defines"], **kwargs)

    def iree_runtime_cc_test(self, deps=[], **kwargs):
        self.cc_test(deps=deps + ["//runtime/src:runtime_defines"], **kwargs)

    def iree_compiler_cc_test(self, deps=[], **kwargs):
        self.cc_test(deps=deps + ["//compiler/src:defs"], **kwargs)

    def iree_runtime_cc_binary(self, deps=[], **kwargs):
        self.cc_binary(deps=deps + ["//runtime/src:runtime_defines"], **kwargs)

    def iree_compiler_cc_binary(self, deps=[], **kwargs):
        self.cc_binary(deps=deps + ["//compiler/src:defs"], **kwargs)


class CustomTargetConverter(bazel_to_cmake_targets.TargetConverter):
    def _initialize(self):
        self._update_target_mappings(
            {
                "//compiler/src:defs": [],
                "//runtime/src:runtime_defines": [],
            }
        )

    def _convert_unmatched_target(self, target: str) -> str:
        """Converts unmatched targets in a repo specific way."""
        # Default rewrite: prefix with "iree::", without pruning the path.
        return ["iree::" + self._convert_to_cmake_path(target)]
