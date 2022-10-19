#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple, Optional, List

import torch
from torch import nn
from torchrec.datasets.utils import Batch
from torchrec.models.dlrm import DLRM
from torchrec.models.deepfm import SimpleDeepFMNN
from torchrec.modules.embedding_modules import EmbeddingBagCollection


class DLRMTrain(nn.Module):
    """
    nn.Module to wrap DLRM model to use with train_pipeline.

    DLRM Recsys model from "Deep Learning Recommendation Model for Personalization and
    Recommendation Systems" (https://arxiv.org/abs/1906.00091). Processes sparse
    features by learning pooled embeddings for each feature. Learns the relationship
    between dense features and sparse features by projecting dense features into the
    same embedding space. Also, learns the pairwise relationships between sparse
    features.

    The module assumes all sparse features have the same embedding dimension
    (i.e, each EmbeddingBagConfig uses the same embedding_dim)

    Args:
        embedding_bag_collection (EmbeddingBagCollection): collection of embedding bags
            used to define SparseArch.
        dense_in_features (int): the dimensionality of the dense input features.
        dense_arch_layer_sizes (list[int]): the layer sizes for the DenseArch.
        over_arch_layer_sizes (list[int]): the layer sizes for the OverArch. NOTE: The
            output dimension of the InteractionArch should not be manually specified
            here.
        dense_device: (Optional[torch.device]).

    Call Args:
        batch: batch used with criteo and random data from torchrec.datasets

    Returns:
        Tuple[loss, Tuple[loss, logits, labels]]

    Example::

        ebc = EmbeddingBagCollection(config=ebc_config)
        model = DLRMTrain(
           embedding_bag_collection=ebc,
           dense_in_features=100,
           dense_arch_layer_sizes=[20],
           over_arch_layer_sizes=[5, 1],
        )
    """

    def __init__(
        self,
        embedding_bag_collection: EmbeddingBagCollection,
        dense_in_features: int,
        dense_arch_layer_sizes: List[int],
        over_arch_layer_sizes: List[int],
        dense_device: Optional[torch.device] = None,
        use_unified: bool = False,
    ) -> None:
        super().__init__()
        self.model = DLRM(
            embedding_bag_collection=embedding_bag_collection,
            dense_in_features=dense_in_features,
            dense_arch_layer_sizes=dense_arch_layer_sizes,
            over_arch_layer_sizes=over_arch_layer_sizes,
            dense_device=dense_device,
            use_unified=use_unified,
        )
        self.loss_fn: nn.Module = nn.BCEWithLogitsLoss()

    def forward(
        self, batch: Batch
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        logits = self.model(batch.dense_features, batch.sparse_features, batch.permu)
        logits = logits.squeeze()
        loss = self.loss_fn(logits, batch.labels.float())
        # print(f"[Note]logits:{logits}\t {logits.detach()}")
        return loss, (loss.detach(), logits.detach(), batch.labels.detach())


class DeepFM(nn.Module):
    def __init__(
        self,
        embedding_bag_collection: EmbeddingBagCollection,
        dense_in_features: int,
        hidden_layer_size: int = 20,
        deep_fm_dimension: int = 5,
        model_name: str = "WDL",
        use_unified: bool = False,
    ) -> None:
        super().__init__()
        self.model = SimpleDeepFMNN(
            num_dense_features=dense_in_features,
            embedding_bag_collection=embedding_bag_collection,
            hidden_layer_size=hidden_layer_size,
            deep_fm_dimension=deep_fm_dimension,
            model_name=model_name,
            use_unified=use_unified,
        )
        self.loss_fn: nn.Module = nn.BCELoss()

    def forward(
        self, batch: Batch
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        logits = self.model(batch.dense_features, batch.sparse_features, batch.permu)
        logits = logits.squeeze()
        loss = self.loss_fn(logits, batch.labels.float())
        return loss, (loss.detach(), logits.detach(), batch.labels.detach())
