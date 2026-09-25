# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Enable with -DIREE_CMAKE_PLUGIN_PATHS=<source>/experimental/dispatch_scheduling.
add_subdirectory(
  "${CMAKE_CURRENT_LIST_DIR}"
  "${IREE_BINARY_DIR}/experimental/dispatch_scheduling"
)
