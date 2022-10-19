#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict, List, Optional

import torch
from torch import nn

# from torchrec import EmbeddingBagCollection, KeyedJaggedTensor
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.modules.deepfm import DeepFM, FactorizationMachine, CrossNet

# from torchrec.sparse.jagged_tensor import KeyedTensor
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor, KeyedTensor

'''
class SparseArch(nn.Module):
    """
    Processes the sparse features of the DeepFMNN model. Does embedding lookups for all
    EmbeddingBag and embedding features of each collection.

    Args:
        embedding_bag_collection (EmbeddingBagCollection): represents a collection of
            pooled embeddings.

    Example::

        eb1_config = EmbeddingBagConfig(
            name="t1", embedding_dim=3, num_embeddings=10, feature_names=["f1"]
        )
        eb2_config = EmbeddingBagConfig(
            name="t2", embedding_dim=4, num_embeddings=10, feature_names=["f2"]
        )

        ebc = EmbeddingBagCollection(tables=[eb1_config, eb2_config])

        #     0       1        2  <-- batch
        # 0   [0,1] None    [2]
        # 1   [3]    [4]    [5,6,7]
        # ^
        # feature
        features = KeyedJaggedTensor.from_offsets_sync(
            keys=["f1", "f2"],
            values=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
            offsets=torch.tensor([0, 2, 2, 3, 4, 5, 8]),
        )

        sparse_arch(features)
    """

    def __init__(
        self, embedding_bag_collection: EmbeddingBagCollection, use_unified: bool
    ) -> None:
        super().__init__()
        self.embedding_bag_collection: EmbeddingBagCollection = embedding_bag_collection
        self.use_unified: bool = use_unified

    def forward(self, features: KeyedJaggedTensor, permu: torch.Tensor) -> KeyedTensor:
        """
        Args:
            features (KeyedJaggedTensor):

        Returns:
            KeyedJaggedTensor: an output KJT of size F * D X B.
        """
        if self.use_unified:
            pass
        else:
            return self.embedding_bag_collection(features)


'''


class SparseArch(nn.Module):
    def __init__(
        self, embedding_bag_collection: EmbeddingBagCollection, use_unified=False
    ) -> None:
        super().__init__()
        self.embedding_bag_collection: EmbeddingBagCollection = embedding_bag_collection
        assert (
            self.embedding_bag_collection.embedding_bag_configs
        ), "Embedding bag collection cannot be empty!"
        self.D: int = self.embedding_bag_collection.embedding_bag_configs[
            0
        ].embedding_dim
        self._sparse_feature_names: List[str] = [
            name
            for conf in embedding_bag_collection.embedding_bag_configs
            for name in conf.feature_names
        ]
        self.F: int = len(self._sparse_feature_names)
        self.use_unified = use_unified

    def forward(self, features: KeyedJaggedTensor, permu: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features (KeyedJaggedTensor): an input tensor of sparse features.

        Returns:
            torch.Tensor: tensor of shape B X F X D.
        """
        sparse_features: KeyedTensor = self.embedding_bag_collection(features)
        if self.use_unified:
            embedding_tensor: torch.Tensor = sparse_features.values()
            embedding_tensor = embedding_tensor[permu.long()].reshape(
                -1, self.F * self.D
            )
            return embedding_tensor

        else:
            B: int = features.stride()

            sparse: Dict[str, torch.Tensor] = sparse_features.to_dict()
            sparse_values: List[torch.Tensor] = []
            for name in self.sparse_feature_names:
                sparse_values.append(sparse[name])

            return torch.cat(sparse_values, dim=1)  # .reshape(B, self.F, self.D)

    @property
    def sparse_feature_names(self) -> List[str]:
        return self._sparse_feature_names


class DenseArch(nn.Module):
    """
    Processes the dense features of DeepFMNN model. Output layer is sized to
    the embedding_dimension of the EmbeddingBagCollection embeddings.

    Args:
        in_features (int): dimensionality of the dense input features.
        hidden_layer_size (int): sizes of the hidden layers in the DenseArch.
        embedding_dim (int): the same size of the embedding_dimension of sparseArch.
        device (torch.device): default compute device.

    Example::

        B = 20
        D = 3
        in_features = 10
        dense_arch = DenseArch(in_features=10, hidden_layer_size=10, embedding_dim=D)
        dense_embedded = dense_arch(torch.rand((B, 10)))
    """

    def __init__(
        self, in_features: int, hidden_layer_size: int, embedding_dim: int,
    ) -> None:
        super().__init__()
        self.model: nn.Module = nn.Sequential(
            nn.Linear(in_features, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, embedding_dim),
            nn.ReLU(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features (torch.Tensor): size B X `num_features`.

        Returns:
            torch.Tensor: an output tensor of size B X D.
        """
        return self.model(features)


class FMInteractionArch(nn.Module):
    """
    Processes the output of both `SparseArch` (sparse_features) and `DenseArch`
    (dense_features) and apply the general DeepFM interaction according to the
    external source of DeepFM paper: https://arxiv.org/pdf/1703.04247.pdf

    The output dimension is expected to be a cat of `dense_features`, D.

    Args:
        fm_in_features (int): the input dimension of `dense_module` in DeepFM. For
            example, if the input embeddings is [randn(3, 2, 3), randn(3, 4, 5)], then
            the `fm_in_features` should be: 2 * 3 + 4 * 5.
        sparse_feature_names (List[str]): length of F.
        deep_fm_dimension (int): output of the deep interaction (DI) in the DeepFM arch.

    Example::

        D = 3
        B = 10
        keys = ["f1", "f2"]
        F = len(keys)
        fm_inter_arch = FMInteractionArch(sparse_feature_names=keys)
        dense_features = torch.rand((B, D))
        sparse_features = KeyedTensor(
            keys=keys,
            length_per_key=[D, D],
            values=torch.rand((B, D * F)),
        )
        cat_fm_output = fm_inter_arch(dense_features, sparse_features)
    """

    def __init__(
        self,
        fm_in_features: int,
        sparse_feature_names: List[str],
        deep_fm_dimension: int,
        has_fm: bool,
    ) -> None:
        super().__init__()
        self.sparse_feature_names: List[str] = sparse_feature_names
        self.deep_fm = DeepFM(
            dense_module=nn.Sequential(
                nn.Linear(fm_in_features, deep_fm_dimension), nn.ReLU(),
            )
        )
        self.has_fm = has_fm
        if self.has_fm:
            self.fm = FactorizationMachine()

    def forward(
        self, dense_features: torch.Tensor, sparse_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            dense_features (torch.Tensor): tensor of size B X D.
            sparse_features (torch.Tensor): tensor of size B X (D*F).

        Returns:
            torch.Tensor: an output tensor of size B X (D + DI + 1).
        """
        if len(self.sparse_feature_names) == 0:
            return dense_features
        """
        tensor_list: List[torch.Tensor] = [dense_features]
        # dense/sparse interaction
        # size B X F
        for feature_name in self.sparse_feature_names:
            tensor_list.append(sparse_features[feature_name])
        """
        interact_tensor = torch.cat([dense_features, sparse_features], dim=1)
        deep_interaction = self.deep_fm(interact_tensor)
        if not self.has_fm:
            return torch.cat([dense_features, deep_interaction], dim=1)
        else:
            fm_interaction = self.fm(interact_tensor)

            def gtr(t: torch.Tensor) -> None:
                return f"[{torch.min(t)},{torch.max(t)}]"

            return torch.cat([dense_features, deep_interaction, fm_interaction], dim=1)


class CrossNetInteractionArch(nn.Module):
    def __init__(
        self,
        fm_in_features: int,
        sparse_feature_names: List[str],
        deep_fm_dimension: int,
        layer_num: int = 2,
    ) -> None:
        super().__init__()
        self.sparse_feature_names: List[str] = sparse_feature_names
        self.deep_fm = DeepFM(
            dense_module=nn.Sequential(
                nn.Linear(fm_in_features, deep_fm_dimension), nn.ReLU(),
            )
        )
        self.crossnet = CrossNet(in_features=fm_in_features, layer_num=layer_num)

    def forward(
        self, dense_features: torch.Tensor, sparse_features: KeyedTensor
    ) -> torch.Tensor:
        """
        Args:
            dense_features (torch.Tensor): tensor of size B X D.
            sparse_features (torch.Tensor): tensor of size B X (D*F).

        Returns:
            torch.Tensor: an output tensor of size B X (D + DI + 1).
        """
        if len(self.sparse_feature_names) == 0:
            return dense_features
        """
        tensor_list: List[torch.Tensor] = [dense_features]
        # dense/sparse interaction
        # size B X F
        for feature_name in self.sparse_feature_names:
            tensor_list.append(sparse_features[feature_name])
        """
        interact_tensor = torch.cat([dense_features, sparse_features], dim=1)
        deep_interaction = self.deep_fm(interact_tensor)
        cross_interaction = self.crossnet(interact_tensor) * 1e-1
        return torch.cat([dense_features, deep_interaction, cross_interaction], dim=1)


class OverArch(nn.Module):
    """
    Final Arch - simple MLP. The output is just one target.

    Args:
        in_features (int): the output dimension of the interaction arch.

    Example::

        B = 20
        over_arch = OverArch()
        logits = over_arch(torch.rand((B, 10)))
    """

    def __init__(self, in_features: int) -> None:
        super().__init__()
        self.model: nn.Module = nn.Sequential(
            nn.Linear(in_features, 1), nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features (torch.Tensor):

        Returns:
            torch.Tensor: an output tensor of size B X 1.
        """
        return self.model(features)


class SimpleDeepFMNN(nn.Module):
    """
    Basic recsys module with DeepFM arch. Processes sparse features by
    learning pooled embeddings for each feature. Learns the relationship between
    dense features and sparse features by projecting dense features into the same
    embedding space. Learns the interaction among those dense and sparse features
    by deep_fm proposed in this paper: https://arxiv.org/pdf/1703.04247.pdf

    The module assumes all sparse features have the same embedding dimension
    (i.e, each `EmbeddingBagConfig` uses the same embedding_dim)

    The following notation is used throughout the documentation for the models:

    * F: number of sparse features
    * D: embedding_dimension of sparse features
    * B: batch size
    * num_features: number of dense features

    Args:
        num_dense_features (int): the number of input dense features.
        embedding_bag_collection (EmbeddingBagCollection): collection of embedding bags
            used to define `SparseArch`.
        hidden_layer_size (int): the hidden layer size used in dense module.
        deep_fm_dimension (int): the output layer size used in `deep_fm`'s deep
            interaction module.

    Example::

        B = 2
        D = 8

        eb1_config = EmbeddingBagConfig(
            name="t1", embedding_dim=D, num_embeddings=100, feature_names=["f1", "f3"]
        )
        eb2_config = EmbeddingBagConfig(
            name="t2",
            embedding_dim=D,
            num_embeddings=100,
            feature_names=["f2"],
        )

        ebc = EmbeddingBagCollection(tables=[eb1_config, eb2_config])
        sparse_nn = SimpleDeepFMNN(
            embedding_bag_collection=ebc, hidden_layer_size=20, over_embedding_dim=5
        )

        features = torch.rand((B, 100))

        #     0       1
        # 0   [1,2] [4,5]
        # 1   [4,3] [2,9]
        # ^
        # feature
        sparse_features = KeyedJaggedTensor.from_offsets_sync(
            keys=["f1", "f3"],
            values=torch.tensor([1, 2, 4, 5, 4, 3, 2, 9]),
            offsets=torch.tensor([0, 2, 4, 6, 8]),
        )

        logits = sparse_nn(
            dense_features=features,
            sparse_features=sparse_features,
        )
        #NOTE: has_fm: Deepfm; not has_fm WDL
    """

    def __init__(
        self,
        num_dense_features: int,
        embedding_bag_collection: EmbeddingBagCollection,
        hidden_layer_size: int,
        deep_fm_dimension: int,
        model_name: str,
        use_unified: bool,
    ) -> None:
        super().__init__()
        assert (
            len(embedding_bag_collection.embedding_bag_configs) > 0
        ), "At least one embedding bag is required"
        for i in range(1, len(embedding_bag_collection.embedding_bag_configs)):
            conf_prev = embedding_bag_collection.embedding_bag_configs[i - 1]
            conf = embedding_bag_collection.embedding_bag_configs[i]
            assert (
                conf_prev.embedding_dim == conf.embedding_dim
            ), "All EmbeddingBagConfigs must have the same dimension"
        embedding_dim: int = embedding_bag_collection.embedding_bag_configs[
            0
        ].embedding_dim

        feature_names = []

        fm_in_features = embedding_dim
        for conf in embedding_bag_collection.embedding_bag_configs:
            for feat in conf.feature_names:
                feature_names.append(feat)
                fm_in_features += conf.embedding_dim

        self.sparse_arch = SparseArch(embedding_bag_collection, use_unified)
        self.dense_arch = DenseArch(
            in_features=num_dense_features,
            hidden_layer_size=hidden_layer_size,
            embedding_dim=embedding_dim,
        )
        inter_out = 0
        if model_name == "WDL" or model_name == "DFM":
            self.inter_arch = FMInteractionArch(
                fm_in_features=fm_in_features,
                sparse_feature_names=feature_names,
                deep_fm_dimension=deep_fm_dimension,
                has_fm=(model_name == "DFM"),
            )
            if model_name == "DFM":
                inter_out = 1

        elif model_name == "DCN":
            self.inter_arch = CrossNetInteractionArch(
                fm_in_features=fm_in_features,
                sparse_feature_names=feature_names,
                deep_fm_dimension=deep_fm_dimension,
            )
            inter_out = fm_in_features

        over_in_features = embedding_dim + deep_fm_dimension + inter_out
        self.over_arch = OverArch(over_in_features)

    def forward(
        self,
        dense_features: torch.Tensor,
        sparse_features: KeyedJaggedTensor,
        permu: torch.Tensor,
    ) -> torch.Tensor:

        """
        def forward(
        self, dense_features: torch.Tensor, sparse_features: KeyedJaggedTensor,
        ) -> torch.Tensor:
        Args:
            dense_features (torch.Tensor): the dense features.
            sparse_features (KeyedJaggedTensor): the sparse features.

        Returns:
            torch.Tensor: logits with size B X 1.
        """
        embedded_dense = self.dense_arch(dense_features)
        embedded_sparse = self.sparse_arch(sparse_features, permu)
        concatenated_dense = self.inter_arch(
            dense_features=embedded_dense, sparse_features=embedded_sparse
        )
        logits = self.over_arch(concatenated_dense)
        return logits
