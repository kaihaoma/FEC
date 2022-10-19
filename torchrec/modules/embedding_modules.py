#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import abc
from typing import List, Dict, Optional

import torch
import torch.nn as nn
from torchrec.modules.embedding_configs import (
    DataType,
    EmbeddingConfig,
    EmbeddingBagConfig,
    pooling_type_to_str,
)
from torchrec.sparse.jagged_tensor import (
    KeyedJaggedTensor,
    JaggedTensor,
    KeyedTensor,
)


class EmbeddingBagCollectionInterface(abc.ABC, nn.Module):
    """
    Interface for `EmbeddingBagCollection`.
    """

    @abc.abstractmethod
    def forward(
        self,
        features: KeyedJaggedTensor,
    ) -> KeyedTensor:
        pass

    @abc.abstractproperty
    def embedding_bag_configs(
        self,
    ) -> List[EmbeddingBagConfig]:
        pass

    @abc.abstractproperty
    def is_weighted(self) -> bool:
        pass


def ebc_get_embedding_names(tables: List[EmbeddingBagConfig]) -> List[str]:
    shared_feature: Dict[str, bool] = {}
    for embedding_config in tables:
        for feature_name in embedding_config.feature_names:
            if feature_name not in shared_feature:
                shared_feature[feature_name] = False
            else:
                shared_feature[feature_name] = True
    embedding_names: List[str] = []
    for embedding_config in tables:
        for feature_name in embedding_config.feature_names:
            if shared_feature[feature_name]:
                embedding_names.append(feature_name + "@" + embedding_config.name)
            else:
                embedding_names.append(feature_name)
    return embedding_names


class EmbeddingBagCollection(EmbeddingBagCollectionInterface):
    """
    EmbeddingBagCollection represents a collection of pooled embeddings (`EmbeddingBags`).

    It processes sparse data in the form of `KeyedJaggedTensor` with values of the form
    [F X B X L] where:

    * F: features (keys)
    * B: batch size
    * L: length of sparse features (jagged)

    and outputs a `KeyedTensor` with values of the form [B * (F * D)] where:

    * F: features (keys)
    * D: each feature's (key's) embedding dimension
    * B: batch size

    Args:
        tables (List[EmbeddingBagConfig]): list of embedding tables.
        is_weighted (bool): whether input `KeyedJaggedTensor` is weighted.
        device (Optional[torch.device]): default compute device.

    Example::

        table_0 = EmbeddingBagConfig(
            name="t1", embedding_dim=3, num_embeddings=10, feature_names=["f1"]
        )
        table_1 = EmbeddingBagConfig(
            name="t2", embedding_dim=4, num_embeddings=10, feature_names=["f2"]
        )

        ebc = EmbeddingBagCollection(tables=[table_0, table_1])

        #        0       1        2  <-- batch
        # "f1"   [0,1] None    [2]
        # "f2"   [3]    [4]    [5,6,7]
        #  ^
        # feature

        features = KeyedJaggedTensor(
            keys=["f1", "f2"],
            values=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
            offsets=torch.tensor([0, 2, 2, 3, 4, 5, 8]),
        )

        pooled_embeddings = ebc(features)
        print(pooled_embeddings.values())
        tensor([[-0.6149,  0.0000, -0.3176],
        [-0.8876,  0.0000, -1.5606],
        [ 1.6805,  0.0000,  0.6810],
        [-1.4206, -1.0409,  0.2249],
        [ 0.1823, -0.4697,  1.3823],
        [-0.2767, -0.9965, -0.1797],
        [ 0.8864,  0.1315, -2.0724]], grad_fn=<TransposeBackward0>)
        print(pooled_embeddings.keys())
        ['f1', 'f2']
        print(pooled_embeddings.offset_per_key())
        tensor([0, 3, 7])
    """

    def __init__(
        self,
        tables: List[EmbeddingBagConfig],
        is_weighted: bool = False,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        torch._C._log_api_usage_once(f"torchrec.modules.{self.__class__.__name__}")
        self._is_weighted = is_weighted
        self.embedding_bags: nn.ModuleDict = nn.ModuleDict()
        self._embedding_bag_configs = tables
        self._lengths_per_embedding: List[int] = []
        table_names = set()
        for embedding_config in tables:
            if embedding_config.name in table_names:
                raise ValueError(f"Duplicate table name {embedding_config.name}")
            table_names.add(embedding_config.name)
            dtype = (
                torch.float32
                if embedding_config.data_type == DataType.FP32
                else torch.float16
            )
            self.embedding_bags[embedding_config.name] = nn.EmbeddingBag(
                num_embeddings=embedding_config.num_embeddings,
                embedding_dim=embedding_config.embedding_dim,
                mode=pooling_type_to_str(embedding_config.pooling),
                device=device,
                include_last_offset=True,
                dtype=dtype,
            )
            if not embedding_config.feature_names:
                embedding_config.feature_names = [embedding_config.name]
            self._lengths_per_embedding.extend(
                len(embedding_config.feature_names) * [embedding_config.embedding_dim]
            )

        self._embedding_names: List[str] = ebc_get_embedding_names(tables)

    def forward(self, features: KeyedJaggedTensor) -> KeyedTensor:
        """
        Args:
            features (KeyedJaggedTensor): KJT of form [F X B X L].

        Returns:
            KeyedTensor
        """

        pooled_embeddings: List[torch.Tensor] = []

        feature_dict = features.to_dict()
        for embedding_config, embedding_bag in zip(
            self._embedding_bag_configs, self.embedding_bags.values()
        ):
            for feature_name in embedding_config.feature_names:
                f = feature_dict[feature_name]
                res = embedding_bag(
                    input=f.values(),
                    offsets=f.offsets(),
                    per_sample_weights=f.weights() if self._is_weighted else None,
                )
                pooled_embeddings.append(res)
        data = torch.cat(pooled_embeddings, dim=1)
        return KeyedTensor(
            keys=self._embedding_names,
            values=data,
            length_per_key=self._lengths_per_embedding,
        )

    @property
    def embedding_bag_configs(self) -> List[EmbeddingBagConfig]:
        return self._embedding_bag_configs

    @property
    def is_weighted(self) -> bool:
        return self._is_weighted


class EmbeddingCollection(nn.Module):
    """
    EmbeddingCollection represents a collection of non-pooled embeddings.

    It processes sparse data in the form of `KeyedJaggedTensor` of the form [F X B X L]
    where:

    * F: features (keys)
    * B: batch size
    * L: length of sparse features (variable)

    and outputs `Dict[feature (key), JaggedTensor]`.
    Each `JaggedTensor` contains values of the form (B * L) X D
    where:

    * B: batch size
    * L: length of sparse features (jagged)
    * D: each feature's (key's) embedding dimension and lengths are of the form L

    Args:
        tables (List[EmbeddingConfig]): list of embedding tables.
        device (Optional[torch.device]): default compute device.
        need_indices (bool): if we need to pass indices to the final lookup result dict

    Example::

        e1_config = EmbeddingConfig(
            name="t1", embedding_dim=2, num_embeddings=10, feature_names=["f1"]
        )
        e2_config = EmbeddingConfig(
            name="t2", embedding_dim=3, num_embeddings=10, feature_names=["f2"]
        )

        ec = EmbeddingCollection(tables=[e1_config, e2_config])

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
        feature_embeddings = ec(features)
        print(feature_embeddings['f2'].values())
        tensor([[-0.2050,  0.5478,  0.6054],
        [ 0.7352,  0.3210, -3.0399],
        [ 0.1279, -0.1756, -0.4130],
        [ 0.7519, -0.4341, -0.0499],
        [ 0.9329, -1.0697, -0.8095]], grad_fn=<EmbeddingBackward>)
    """

    def __init__(  # noqa C901
        self,
        tables: List[EmbeddingConfig],
        device: Optional[torch.device] = None,
        need_indices: bool = False,
    ) -> None:
        super().__init__()
        torch._C._log_api_usage_once(f"torchrec.modules.{self.__class__.__name__}")
        self.embeddings: nn.ModuleDict = nn.ModuleDict()
        self.embedding_configs = tables
        self.embedding_dim: int = -1
        self.need_indices: bool = need_indices
        table_names = set()
        shared_feature: Dict[str, bool] = {}
        for config in tables:
            if config.name in table_names:
                raise ValueError(f"Duplicate table name {config.name}")
            table_names.add(config.name)
            self.embedding_dim = (
                config.embedding_dim if self.embedding_dim < 0 else self.embedding_dim
            )
            if self.embedding_dim != config.embedding_dim:
                raise ValueError(
                    "All tables in a EmbeddingCollection are required to have same embedding dimension."
                )
            dtype = (
                torch.float32 if config.data_type == DataType.FP32 else torch.float16
            )
            self.embeddings[config.name] = nn.Embedding(
                num_embeddings=config.num_embeddings,
                embedding_dim=config.embedding_dim,
                device=device,
                dtype=dtype,
            )
            if not config.feature_names:
                config.feature_names = [config.name]
            for feature_name in config.feature_names:
                if feature_name not in shared_feature:
                    shared_feature[feature_name] = False
                else:
                    shared_feature[feature_name] = True

        self.embedding_names_by_table: List[List[str]] = []
        for config in tables:
            embedding_names = []
            for feature_name in config.feature_names:
                if shared_feature[feature_name]:
                    embedding_names.append(feature_name + "@" + config.name)
                else:
                    embedding_names.append(feature_name)
            self.embedding_names_by_table.append(embedding_names)

    def forward(
        self,
        features: KeyedJaggedTensor,
    ) -> Dict[str, JaggedTensor]:
        """
        Args:
            features (KeyedJaggedTensor): KJT of form [F X B X L].

        Returns:
            Dict[str, JaggedTensor]
        """

        feature_embeddings: Dict[str, JaggedTensor] = {}
        jt_dict: Dict[str, JaggedTensor] = features.to_dict()
        for config, embedding_names, emb_module in zip(
            self.embedding_configs,
            self.embedding_names_by_table,
            self.embeddings.values(),
        ):
            for feature_name, embedding_name in zip(
                config.feature_names, embedding_names
            ):
                f = jt_dict[feature_name]
                lookup = emb_module(
                    input=f.values(),
                )
                feature_embeddings[embedding_name] = JaggedTensor(
                    values=lookup,
                    lengths=f.lengths(),
                    weights=f.values() if self.need_indices else None,
                )
        return feature_embeddings
