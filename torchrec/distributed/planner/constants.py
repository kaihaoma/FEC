#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

from torchrec.distributed.embedding_types import EmbeddingComputeKernel

MAX_SIZE: int = (1 << 63) - 1

# INTRA_NODE_BANDWIDTH: float = 600 * 1024 * 1024 * 1024 / 1000  # bytes/ms
INTRA_NODE_BANDWIDTH: float = 250 * 1024 * 1024 * 1024 / 1000
CROSS_NODE_BANDWIDTH: float = 25 * 1024 * 1024 * 1024 / 1000  # bytes/ms

MIN_CW_DIM: int = 128
POOLING_FACTOR: float = 1.0

BIGINT_DTYPE: int = 8

HBM_CAP: int = 16 * 1024 * 1024 * 1024  # 16 GB
DDR_CAP: int = 378 * 1024 * 1024 * 1024  # 360 GB
DDR_MEM_BW: float = 51 * 1024 * 1024 * 1024 / 1000  # bytes/ms
HBM_MEM_BW: float = 897 * 1024 * 1024 * 1024 / 1000  # bytes/ms
UVM_CACHING_RATIO: float = 0.2
BATCH_SIZE: int = 512

BATCHED_COPY_PERF_FACTOR: float = 2.455  # empirical studies
FULL_BLOCK_EMB_DIM: int = 128  # FBGEMM Kernel, 32 threads X 4D-Vector
HALF_BLOCK_PENALTY: float = 1.15  # empirical studies
QUARTER_BLOCK_PENALTY: float = 1.75  # empirical studies
BWD_COMPUTE_MULTIPLIER: float = 2  # empirical studies
WEIGHTED_KERNEL_MULTIPLIER: float = 1.1  # empirical studies
DP_ELEMENTWISE_KERNELS_PERF_FACTOR: float = 9.22  # empirical studies
ALLREDUCE_MEMORY_MULTIPLIER: float = 1.875  # NVIDIA presentation: https://images.nvidia.com/events/sc15/pdfs/NCCL-Woolley.pdf


def kernel_bw_lookup(
    compute_device: str, compute_kernel: str, caching_ratio: Optional[float] = None,
) -> Optional[float]:
    """
    Calculates the device bandwidth based on given compute device, compute kernel, and
    caching ratio.

    Args:
        compute_kernel (str): compute kernel.
        compute_device (str): compute device.
        caching_ratio (Optional[float]): caching ratio used to determine device bandwidth
            if UVM caching is enabled.

    Returns:
        float: the device bandwidth.
    """
    caching_ratio = caching_ratio if caching_ratio else UVM_CACHING_RATIO
    lookup = {
        # CPU
        ("cpu", EmbeddingComputeKernel.DENSE.value): 0.35 * DDR_MEM_BW,
        ("cpu", EmbeddingComputeKernel.SPARSE.value): 0.35 * DDR_MEM_BW,
        ("cpu", EmbeddingComputeKernel.BATCHED_DENSE.value): 0.5 * DDR_MEM_BW,
        ("cpu", EmbeddingComputeKernel.BATCHED_FUSED.value): 1 * DDR_MEM_BW,
        ("cpu", EmbeddingComputeKernel.BATCHED_QUANT.value): 1 * DDR_MEM_BW,
        # CUDA
        ("cuda", EmbeddingComputeKernel.DENSE.value): 0.35 * HBM_MEM_BW,
        ("cuda", EmbeddingComputeKernel.SPARSE.value): 0.35 * HBM_MEM_BW,
        ("cuda", EmbeddingComputeKernel.BATCHED_DENSE.value): 0.5 * HBM_MEM_BW,
        ("cuda", EmbeddingComputeKernel.BATCHED_FUSED.value): 1 * HBM_MEM_BW,
        ("cuda", EmbeddingComputeKernel.BATCHED_FUSED_UVM.value): DDR_MEM_BW / 100,
        ("cuda", EmbeddingComputeKernel.BATCHED_FUSED_UVM_CACHING.value): (
            caching_ratio * HBM_MEM_BW + (1 - caching_ratio) * DDR_MEM_BW
        )
        / 100,
        ("cuda", EmbeddingComputeKernel.BATCHED_QUANT.value): 1 * HBM_MEM_BW,
        ("cuda", EmbeddingComputeKernel.BATCHED_QUANT_UVM.value): DDR_MEM_BW / 100,
        ("cuda", EmbeddingComputeKernel.BATCHED_QUANT_UVM_CACHING.value): (
            caching_ratio * HBM_MEM_BW + (1 - caching_ratio) * DDR_MEM_BW
        )
        / 100,
    }
    return lookup.get((compute_device, compute_kernel))
