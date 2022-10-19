#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
from collections import OrderedDict
from itertools import accumulate
from typing import (
    List,
    Dict,
    Optional,
    Type,
    Any,
    Mapping,
    Union,
    Tuple,
    Iterator,
    Set,
)

import torch
from torch import Tensor
from torch import nn
from torch.nn.modules.module import _IncompatibleKeys
from torchrec.distributed.embedding_sharding import (
    EmbeddingSharding,
    SparseFeaturesListAwaitable,
)
from torchrec.distributed.embedding_types import (
    UnifiedEmbeddingConfig,
    SparseFeatures,
    BaseEmbeddingSharder,
    EmbeddingComputeKernel,
    SparseFeaturesList,
)
from torchrec.distributed.sharding.cw_sharding import (
    CwPooledEmbeddingSharding,
    UnifiedCwEmbeddingSharding,
)
from torchrec.distributed.sharding.dp_sharding import (
    DpPooledEmbeddingSharding,
    UnifiedDpEmbeddingSharding,
)
from torchrec.distributed.sharding.rw_sharding import RwPooledEmbeddingSharding
from torchrec.distributed.sharding.tw_sharding import TwPooledEmbeddingSharding
from torchrec.distributed.sharding.twcw_sharding import TwCwPooledEmbeddingSharding
from torchrec.distributed.sharding.twrw_sharding import TwRwPooledEmbeddingSharding
from torchrec.distributed.types import (
    NoWait,
    Awaitable,
    EnumerableShardingSpec,
    LazyAwaitable,
    ParameterSharding,
    ShardedModule,
    ShardingType,
    ShardedModuleContext,
    ShardedTensor,
    ShardingEnv,
)
from torchrec.distributed.utils import append_prefix
from torchrec.modules.embedding_configs import EmbeddingTableConfig, PoolingType
from torchrec.modules.embedding_modules import (
    EmbeddingBagCollection,
    EmbeddingBagCollectionInterface,
)
from torchrec.optim.fused import FusedOptimizerModule
from torchrec.optim.keyed import KeyedOptimizer, CombinedOptimizer
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor, KeyedTensor


def replace_placement_with_meta_device(
    embedding_configs: List[
        Tuple[EmbeddingTableConfig, ParameterSharding, torch.Tensor]
    ]
) -> None:
    """Placement device and tensor device could be unmatched in some
    scenarios, e.g. passing meta device to DMP and passing cuda
    to EmbeddingShardingPlanner. We need to make device consistent
    after getting sharding planner.
    """
    for config in embedding_configs:
        sharding_spec = config[1].sharding_spec
        if sharding_spec is None:
            continue
        if isinstance(sharding_spec, EnumerableShardingSpec):
            for shard_metadata in sharding_spec.shards:
                placement = shard_metadata.placement
                if isinstance(placement, str):
                    placement = torch.distributed._remote_device(placement)
                assert isinstance(placement, torch.distributed._remote_device)
                placement._device = torch.device("meta")
                shard_metadata.placement = placement
        else:
            # We only support EnumerableShardingSpec at present.
            raise RuntimeError(
                f"Unsupported ShardingSpec {type(sharding_spec)} with meta device"
            )


def create_embedding_bag_sharding(
    sharding_type: str,
    embedding_configs: List[
        Tuple[EmbeddingTableConfig, ParameterSharding, torch.Tensor]
    ],
    env: ShardingEnv,
    device: Optional[torch.device] = None,
    permute_embeddings: bool = False,
) -> EmbeddingSharding[SparseFeatures, torch.Tensor]:
    if device is not None and device.type == "meta":
        replace_placement_with_meta_device(embedding_configs)
    if sharding_type == ShardingType.TABLE_WISE.value:
        return TwPooledEmbeddingSharding(embedding_configs, env, device)
    elif sharding_type == ShardingType.ROW_WISE.value:
        return RwPooledEmbeddingSharding(embedding_configs, env, device)
    elif sharding_type == ShardingType.DATA_PARALLEL.value:
        return DpPooledEmbeddingSharding(embedding_configs, env, device)
    elif sharding_type == ShardingType.TABLE_ROW_WISE.value:
        return TwRwPooledEmbeddingSharding(embedding_configs, env, device)
    elif sharding_type == ShardingType.COLUMN_WISE.value:
        return CwPooledEmbeddingSharding(
            embedding_configs, env, device, permute_embeddings=permute_embeddings
        )
    elif sharding_type == ShardingType.TABLE_COLUMN_WISE.value:
        return TwCwPooledEmbeddingSharding(
            embedding_configs, env, device, permute_embeddings=permute_embeddings
        )
    else:
        raise ValueError(f"Sharding type not supported {sharding_type}")


def filter_state_dict(
    state_dict: "OrderedDict[str, torch.Tensor]", name: str
) -> "OrderedDict[str, torch.Tensor]":
    rtn_dict = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith(name):
            # + 1 to length is to remove the '.' after the key
            rtn_dict[key[len(name) + 1 :]] = value
    return rtn_dict


def create_embedding_configs_by_sharding(
    module: EmbeddingBagCollectionInterface,
    table_name_to_parameter_sharding: Dict[str, ParameterSharding],
    prefix: str,
) -> Dict[str, List[Tuple[EmbeddingTableConfig, ParameterSharding, torch.Tensor]]]:
    shared_feature: Dict[str, bool] = {}
    for embedding_config in module.embedding_bag_configs:
        if not embedding_config.feature_names:
            embedding_config.feature_names = [embedding_config.name]
        for feature_name in embedding_config.feature_names:
            if feature_name not in shared_feature:
                shared_feature[feature_name] = False
            else:
                shared_feature[feature_name] = True

    sharding_type_to_embedding_configs: Dict[
        str, List[Tuple[EmbeddingTableConfig, ParameterSharding, torch.Tensor]]
    ] = {}
    state_dict = module.state_dict()
    for config in module.embedding_bag_configs:
        table_name = config.name
        assert table_name in table_name_to_parameter_sharding
        parameter_sharding = table_name_to_parameter_sharding[table_name]
        if parameter_sharding.compute_kernel not in [
            kernel.value for kernel in EmbeddingComputeKernel
        ]:
            raise ValueError(
                f"Compute kernel not supported {parameter_sharding.compute_kernel}"
            )
        embedding_names: List[str] = []
        for feature_name in config.feature_names:
            if shared_feature[feature_name]:
                embedding_names.append(feature_name + "@" + config.name)
            else:
                embedding_names.append(feature_name)

        param_name = prefix + table_name + ".weight"
        assert param_name in state_dict
        param = state_dict[param_name]

        if parameter_sharding.sharding_type not in sharding_type_to_embedding_configs:
            sharding_type_to_embedding_configs[parameter_sharding.sharding_type] = []
        sharding_type_to_embedding_configs[parameter_sharding.sharding_type].append(
            (
                EmbeddingTableConfig(
                    num_embeddings=config.num_embeddings,
                    embedding_dim=config.embedding_dim,
                    name=config.name,
                    data_type=config.data_type,
                    feature_names=copy.deepcopy(config.feature_names),
                    pooling=config.pooling,
                    is_weighted=module.is_weighted,
                    has_feature_processor=False,
                    embedding_names=embedding_names,
                    weight_init_max=config.weight_init_max,
                    weight_init_min=config.weight_init_min,
                ),
                parameter_sharding,
                param,
            )
        )
    return sharding_type_to_embedding_configs


class UnifiedEmbeddingBagCollectionAwaitable(LazyAwaitable[KeyedTensor]):
    def __init__(self, awaitables: List[Awaitable[torch.Tensor]]) -> None:
        super().__init__()
        self._awaitables = awaitables

    def _wait_impl(self) -> KeyedTensor:

        embeddings = [w.wait() for w in self._awaitables]
        if len(embeddings) == 1:
            assert False
            embeddings = embeddings[0]
        else:
            embeddings = torch.cat(embeddings, dim=0)
        return KeyedTensor(
            keys="", length_per_key=embeddings.numel(), values=embeddings
        )


class EmbeddingBagCollectionAwaitable(LazyAwaitable[KeyedTensor]):
    def __init__(
        self,
        awaitables: List[Awaitable[torch.Tensor]],
        embedding_dims: List[int],
        embedding_names: List[str],
    ) -> None:
        super().__init__()
        self._awaitables = awaitables
        self._embedding_dims = embedding_dims
        self._embedding_names = embedding_names

    def _wait_impl(self) -> KeyedTensor:
        embeddings = [w.wait() for w in self._awaitables]
        if len(embeddings) == 1:
            embeddings = embeddings[0]
        else:
            embeddings = torch.cat(embeddings, dim=1)
        return KeyedTensor(
            keys=self._embedding_names,
            length_per_key=self._embedding_dims,
            values=embeddings,
            key_dim=1,
        )


class ShardedEmbeddingBagCollection(
    ShardedModule[SparseFeaturesList, List[torch.Tensor], KeyedTensor],
    FusedOptimizerModule,
):
    """
    Sharded implementation of EmbeddingBagCollection.
    This is part of the public API to allow for manual data dist pipelining.
    """

    def __init__(
        self,
        module: EmbeddingBagCollectionInterface,
        table_name_to_parameter_sharding: Dict[str, ParameterSharding],
        env: ShardingEnv,
        fused_params: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
        use_unified: bool = False,
        unified_cw_config: Optional[UnifiedEmbeddingConfig] = None,
        unified_dp_config: Optional[UnifiedEmbeddingConfig] = None,
        use_prefetch: bool = False,
    ) -> None:
        super().__init__()
        self._lookups: nn.ModuleList = nn.ModuleList()
        self._use_unified: bool = use_unified
        self._use_prefetch: bool = use_prefetch
        self._sharding_type_to_sharding: Dict[str, Any] = {}
        self._batch_size_per_rank: Optional[List[int]] = None
        self._world_size: int = env.world_size
        self._rank: int = env.rank
        if not self._use_unified:
            sharding_type_to_embedding_configs = create_embedding_configs_by_sharding(
                module, table_name_to_parameter_sharding, "embedding_bags."
            )
            self._sharding_type_to_sharding = {
                sharding_type: create_embedding_bag_sharding(
                    sharding_type,
                    embedding_confings,
                    env,
                    device,
                    permute_embeddings=True,
                )
                for sharding_type, embedding_confings in sharding_type_to_embedding_configs.items()
            }

            self._create_lookups(fused_params)
        else:
            assert self._use_unified and unified_cw_config and unified_dp_config
            self._sharding_type_to_sharding = {
                ShardingType.COLUMN_WISE.value: UnifiedCwEmbeddingSharding(
                    unified_cw_config=unified_cw_config, env=env, device=device,
                ),
                ShardingType.DATA_PARALLEL.value: UnifiedDpEmbeddingSharding(
                    unified_dp_config=unified_dp_config, env=env, device=device,
                ),
            }
            self._create_unified_lookups(fused_params)
            self._cold_entries_sharding_type: str = "rw"

        self._is_weighted: bool = module.is_weighted
        self._device = device
        self._input_dists = nn.ModuleList()
        self._output_dists: nn.ModuleList = nn.ModuleList()
        self._embedding_names: List[str] = []
        self._embedding_dims: List[int] = []
        self._feature_splits: List[int] = []
        self._features_order: List[int] = []

        # check all the sharding:
        print(
            f"[Note][ShardedEmbeddingBagCollection]: ShardedEmbeddingBagCollection:{self._sharding_type_to_sharding}"
        )
        # forward pass flow control
        self._has_uninitialized_input_dist: bool = True
        self._has_uninitialized_output_dist: bool = True
        self._has_features_permute: bool = True
        # Get all fused optimizers and combine them.
        optims = []
        for lookup in self._lookups:
            for _, module in lookup.named_modules():
                if isinstance(module, FusedOptimizerModule):
                    # modify param keys to match EmbeddingBagCollection
                    params: Mapping[str, Union[torch.Tensor, ShardedTensor]] = {}
                    for param_key, weight in module.fused_optimizer.params.items():
                        # pyre-fixme[16]: `Mapping` has no attribute `__setitem__`.
                        params["embedding_bags." + param_key] = weight
                    module.fused_optimizer.params = params
                    optims.append(("", module.fused_optimizer))

        for name, param in self.named_parameters():
            self.register_parameter("unified_dp", param)

        self._optim: CombinedOptimizer = CombinedOptimizer(optims)

    def _create_input_dist(self, input_feature_names: List[str],) -> None:
        feature_names: List[str] = []
        for sharding in self._sharding_type_to_sharding.values():
            self._input_dists.append(sharding.create_input_dist())
            feature_names.extend(
                sharding.id_score_list_feature_names()
                if self._is_weighted
                else sharding.id_list_feature_names()
            )
            self._feature_splits.append(
                len(
                    sharding.id_score_list_feature_names()
                    if self._is_weighted
                    else sharding.id_list_feature_names()
                )
            )

        if feature_names == input_feature_names:
            self._has_features_permute = False
        else:
            for f in feature_names:
                self._features_order.append(input_feature_names.index(f))
            self.register_buffer(
                "_features_order_tensor",
                torch.tensor(
                    self._features_order, device=self._device, dtype=torch.int32
                ),
            )

    def _create_lookups(self, fused_params: Optional[Dict[str, Any]],) -> None:
        for sharding in self._sharding_type_to_sharding.values():
            self._lookups.append(sharding.create_lookup(fused_params=fused_params))

    def _create_output_dist(self) -> None:
        for sharding in self._sharding_type_to_sharding.values():
            self._output_dists.append(sharding.create_output_dist(device=self._device))
            self._embedding_names.extend(sharding.embedding_names())
            self._embedding_dims.extend(sharding.embedding_dims())

    # TODO unified_input_dist, compute and output_dist
    def _create_unified_input_dist(self) -> None:
        for sharding in self._sharding_type_to_sharding.values():
            self._input_dists.append(sharding.create_unified_input_dist())

    def _create_unified_lookups(self, fused_params: Optional[Dict[str, Any]],) -> None:
        for sharding in self._sharding_type_to_sharding.values():
            self._lookups.append(
                sharding.create_unified_lookup(fused_params=fused_params)
            )

    def _create_unified_output_dist(self) -> None:
        for sharding in self._sharding_type_to_sharding.values():
            self._output_dists.append(
                sharding.create_unified_output_dist(device=self._device)
            )

    def unified_input_dist(
        self, ctx: ShardedModuleContext, features: KeyedJaggedTensor,
    ) -> Awaitable[SparseFeaturesList]:
        if self._has_uninitialized_input_dist:
            self._create_unified_input_dist()
            self._has_uninitialized_input_dist = False

        assert not self._is_weighted

        with torch.no_grad():
            list_jct = features.to_list_kjt(
                world_size=self._world_size,
                cold_entries_partition_type=self._cold_entries_sharding_type,
                use_prefetch=self._use_prefetch,
            )

            awaitables = []
            for module, features_by_shard in zip(self._input_dists, list_jct):
                all2all_lengths = module(
                    SparseFeatures(
                        id_list_features=None
                        if self._is_weighted
                        else features_by_shard,
                        id_score_list_features=features_by_shard
                        if self._is_weighted
                        else None,
                    )
                )

                awaitables.append(all2all_lengths.wait())

            if not self._use_prefetch or self._has_uninitialized_output_dist:

                return SparseFeaturesListAwaitable(awaitables)

        if not self._has_uninitialized_output_dist:
            sparse_features_list = SparseFeaturesListAwaitable(awaitables).wait()
            cold_entries_kjt = sparse_features_list[0].id_list_features

            length_per_worker: List[int] = cold_entries_kjt._length_per_key
            offset_per_worker: List[int] = cold_entries_kjt._offset_per_key

            n_prefetch_entries: int = sum(
                [length_per_worker[2 * wid] for wid in range(self._world_size)]
            )
            n_overlap_entries: int = sum(
                [length_per_worker[2 * wid + 1] for wid in range(self._world_size)]
            )

            # NOTE: special case: no prefetch entries
            if n_prefetch_entries == 0:
                return NoWait(sparse_features_list)

            cold_entries_tensor: torch.Tensor = cold_entries_kjt.values()
            accu_length_per_worker: List[int] = [0] + list(
                accumulate(length_per_worker)
            )
            assert (
                cold_entries_tensor.numel() == accu_length_per_worker[-1]
            ), f"cold entries len:{cold_entries_tensor.numel()}, provide len:{accu_length_per_worker[-1]}"

            prefetch_entries_tensor: torch.Tensor = torch.cat(
                [
                    cold_entries_tensor[
                        slice(
                            accu_length_per_worker[2 * wid],
                            accu_length_per_worker[2 * wid + 1],
                        )
                    ]
                    for wid in range(self._world_size)
                ]
            )
            overlap_entries_tensor: torch.Tensor = torch.cat(
                [
                    cold_entries_tensor[
                        slice(
                            accu_length_per_worker[2 * wid + 1],
                            accu_length_per_worker[2 * wid + 2],
                        )
                    ]
                    for wid in range(self._world_size)
                ]
            )
            prefetch_entries = SparseFeatures(
                id_list_features=KeyedJaggedTensor(
                    keys=["prefetch"],
                    values=prefetch_entries_tensor,
                    offsets=cold_entries_kjt.offsets()[: n_prefetch_entries + 1],
                    length_per_key=[
                        length_per_worker[2 * wid] for wid in range(self._world_size)
                    ],
                    offset_per_key=[
                        offset_per_worker[2 * wid] for wid in range(self._world_size)
                    ],
                )
            )
            overlap_entries = SparseFeatures(
                id_list_features=KeyedJaggedTensor(
                    keys=["overlap"],
                    values=overlap_entries_tensor,
                    offsets=cold_entries_kjt.offsets()[: n_overlap_entries + 1],
                    length_per_key=[
                        length_per_worker[2 * wid + 1]
                        for wid in range(self._world_size)
                    ],
                    offset_per_key=[
                        offset_per_worker[2 * wid + 1]
                        for wid in range(self._world_size)
                    ],
                )
            )

            # prefetch_result: torch.Tensor = self._output_dists[0]
            # cw output dist:
            # param: 1: SparseFeatures 2: batch_size_per_rank: List[int]
            prefetch_result: torch.Tensor = self._output_dists[0](
                self._lookups[0](prefetch_entries),
                prefetch_entries.id_list_features._length_per_key,
                prefetch_entries.id_list_features._offset_per_key,
            )
            new_sparse_features_list = SparseFeaturesList(
                [overlap_entries, sparse_features_list[1], prefetch_result]
            )

            return NoWait(new_sparse_features_list)

    def unified_compute_and_output_dist(
        self, ctx: ShardedModuleContext, input: SparseFeaturesList
    ) -> LazyAwaitable[KeyedTensor]:
        # NOTE: batch_size_per_rank: List[int] should be provided
        if self._has_uninitialized_output_dist:
            self._create_unified_output_dist()
            self._has_uninitialized_output_dist = False

        len_lpk = len(input[0].id_list_features._length_per_key)
        len_opk = len(input[0].id_list_features._offset_per_key)

        if self._use_prefetch:

            if len_lpk > self._world_size:

                input[0].id_list_features._length_per_key = [
                    input[0].id_list_features._length_per_key[wid * 2 + 1]
                    for wid in range(self._world_size)
                ]

            if len_opk > self._world_size:

                input[0].id_list_features._offset_per_key = [
                    input[0].id_list_features._offset_per_key[wid * 2 + 1]
                    for wid in range(self._world_size)
                ]
        else:
            assert (
                len_lpk == self._world_size * 2
            ), f"len_lpk:{len_lpk} != {self._world_size * 2}"
            assert (
                len_opk == self._world_size * 2
            ), f"len_opk:{len_opk} != {self._world_size * 2}"
            input[0].id_list_features._length_per_key = [
                input[0].id_list_features._length_per_key[wid * 2]
                + input[0].id_list_features._length_per_key[wid * 2 + 1]
                for wid in range(self._world_size)
            ]

            input[0].id_list_features._offset_per_key = [
                input[0].id_list_features._offset_per_key[wid * 2]
                + input[0].id_list_features._offset_per_key[wid * 2 + 1]
                for wid in range(self._world_size)
            ]

        awaitables = [
            dist(
                lookup(features),
                features.id_list_features._length_per_key,  # rw & cw, None for dp (output_batch_size_per_rank)
                features.id_list_features._offset_per_key,  # rw, None for cw, dp (input_batch_size_per_rank)
            )
            for lookup, dist, features in zip(self._lookups, self._output_dists, input,)
        ]

        if len(input) > 2:
            awaitables.insert(0, NoWait(input[2]))

        return UnifiedEmbeddingBagCollectionAwaitable(awaitables=awaitables,)

    # End of Unified code
    # Override forward Function, used in connect(once)
    def forward(self, *input, **kwargs) -> LazyAwaitable[KeyedTensor]:

        if self._use_unified:
            ctx = self.create_context()
            unified_dist_input = self.unified_input_dist(ctx, *input, **kwargs).wait()
            return self.unified_compute_and_output_dist(ctx, unified_dist_input)

        else:
            ctx = self.create_context()
            dist_input = self.input_dist(ctx, *input, **kwargs).wait()
            return self.compute_and_output_dist(ctx, dist_input)

    # pyre-ignore [14]
    def input_dist(
        self, ctx: ShardedModuleContext, features: KeyedJaggedTensor,
    ) -> Awaitable[SparseFeaturesList]:
        if self._has_uninitialized_input_dist:
            self._create_input_dist(features.keys())
            self._has_uninitialized_input_dist = False

        with torch.no_grad():
            if self._has_features_permute:
                features = features.permute(
                    self._features_order,
                    # pyre-ignore [6]
                    self._features_order_tensor,
                )
            features_by_shards = features.split(self._feature_splits,)
            awaitables = []
            for module, features_by_shard in zip(self._input_dists, features_by_shards):
                all2all_lengths = module(
                    SparseFeatures(
                        id_list_features=None
                        if self._is_weighted
                        else features_by_shard,
                        id_score_list_features=features_by_shard
                        if self._is_weighted
                        else None,
                    )
                )
                awaitables.append(all2all_lengths.wait())
            return SparseFeaturesListAwaitable(awaitables)

    def compute(
        self, ctx: ShardedModuleContext, dist_input: SparseFeaturesList,
    ) -> List[torch.Tensor]:
        return [lookup(features) for lookup, features in zip(self._lookups, dist_input)]

    def output_dist(
        self, ctx: ShardedModuleContext, output: List[torch.Tensor],
    ) -> LazyAwaitable[KeyedTensor]:
        if self._has_uninitialized_output_dist:
            self._create_output_dist()
            self._has_uninitialized_output_dist = False
        return EmbeddingBagCollectionAwaitable(
            awaitables=[
                dist(embeddings) for dist, embeddings in zip(self._output_dists, output)
            ],
            embedding_dims=self._embedding_dims,
            embedding_names=self._embedding_names,
        )

    def compute_and_output_dist(
        self, ctx: ShardedModuleContext, input: SparseFeaturesList
    ) -> LazyAwaitable[KeyedTensor]:
        if self._has_uninitialized_output_dist:
            self._create_output_dist()
            self._has_uninitialized_output_dist = False
        return EmbeddingBagCollectionAwaitable(
            awaitables=[
                dist(lookup(features))
                for lookup, dist, features in zip(
                    self._lookups, self._output_dists, input,
                )
            ],
            embedding_dims=self._embedding_dims,
            embedding_names=self._embedding_names,
        )

    # pyre-fixme[14]: `state_dict` overrides method defined in `Module` inconsistently.
    def state_dict(
        self,
        destination: Optional[Dict[str, Any]] = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> Dict[str, Any]:
        if destination is None:
            destination = OrderedDict()
            # pyre-ignore [16]
            destination._metadata = OrderedDict()
        for lookup in self._lookups:
            lookup.state_dict(destination, prefix + "embedding_bags.", keep_vars)
        return destination

    def named_modules(
        self,
        memo: Optional[Set[nn.Module]] = None,
        prefix: str = "",
        remove_duplicate: bool = True,
    ) -> Iterator[Tuple[str, nn.Module]]:
        yield from [(prefix, self)]

    def named_parameters(
        self, prefix: str = "", recurse: bool = True
    ) -> Iterator[Tuple[str, nn.Parameter]]:
        for lookup in self._lookups:
            yield from lookup.named_parameters(
                append_prefix(prefix, "embedding_bags"), recurse
            )

    def sharded_parameter_names(self, prefix: str = "") -> Iterator[str]:
        for lookup, sharding_type in zip(
            self._lookups, self._sharding_type_to_sharding.keys()
        ):
            if sharding_type == ShardingType.DATA_PARALLEL.value:
                continue
            for name, _ in lookup.named_parameters(
                append_prefix(prefix, "embedding_bags")
            ):
                yield name

    def named_buffers(
        self, prefix: str = "", recurse: bool = True
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        for lookup in self._lookups:
            yield from lookup.named_buffers(
                append_prefix(prefix, "embedding_bags"), recurse
            )

    def load_state_dict(
        self, state_dict: "OrderedDict[str, torch.Tensor]", strict: bool = True,
    ) -> _IncompatibleKeys:
        missing_keys = []
        unexpected_keys = []
        for lookup in self._lookups:
            missing, unexpected = lookup.load_state_dict(
                filter_state_dict(state_dict, "embedding_bags"), strict,
            )
            missing_keys.extend(missing)
            unexpected_keys.extend(unexpected)
        return _IncompatibleKeys(
            missing_keys=missing_keys, unexpected_keys=unexpected_keys
        )

    def sparse_grad_parameter_names(
        self, destination: Optional[List[str]] = None, prefix: str = "",
    ) -> List[str]:
        destination = [] if destination is None else destination
        for lookup in self._lookups:
            lookup.sparse_grad_parameter_names(
                destination, append_prefix(prefix, "embedding_bags")
            )
        return destination

    @property
    def fused_optimizer(self) -> KeyedOptimizer:
        return self._optim


class EmbeddingBagCollectionSharder(BaseEmbeddingSharder[EmbeddingBagCollection]):
    """
    This implementation uses non-fused `EmbeddingBagCollection`
    """

    def shard(
        self,
        module: EmbeddingBagCollection,
        params: Dict[str, ParameterSharding],
        env: ShardingEnv,
        device: Optional[torch.device] = None,
        use_unified: bool = False,
        use_prefetch: bool = False,
        unified_cw_config: Optional[UnifiedEmbeddingConfig] = None,
        unified_dp_config: Optional[UnifiedEmbeddingConfig] = None,
    ) -> ShardedEmbeddingBagCollection:
        return ShardedEmbeddingBagCollection(
            module=module,
            table_name_to_parameter_sharding=params,
            env=env,
            fused_params=self.fused_params,
            device=device,
            use_unified=use_unified,
            use_prefetch=use_prefetch,
            unified_cw_config=unified_cw_config,
            unified_dp_config=unified_dp_config,
        )

    def shardable_parameters(
        self, module: EmbeddingBagCollection
    ) -> Dict[str, nn.Parameter]:
        return {
            name.split(".")[0]: param
            for name, param in module.embedding_bags.named_parameters()
        }

    @property
    def module_type(self) -> Type[EmbeddingBagCollection]:
        return EmbeddingBagCollection


class EmbeddingAwaitable(LazyAwaitable[torch.Tensor]):
    def __init__(self, awaitable: Awaitable[torch.Tensor],) -> None:
        super().__init__()
        self._awaitable = awaitable

    def _wait_impl(self) -> torch.Tensor:
        embedding = self._awaitable.wait()
        return embedding


class ShardedEmbeddingBag(
    ShardedModule[SparseFeatures, torch.Tensor, torch.Tensor,], FusedOptimizerModule,
):
    """
    Sharded implementation of `nn.EmbeddingBag`.
    This is part of the public API to allow for manual data dist pipelining.
    """

    def __init__(
        self,
        module: nn.EmbeddingBag,
        table_name_to_parameter_sharding: Dict[str, ParameterSharding],
        env: ShardingEnv,
        fused_params: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()

        assert (
            len(table_name_to_parameter_sharding) == 1
        ), "expect 1 table, but got len(table_name_to_parameter_sharding)"
        assert module.mode == "sum", "ShardedEmbeddingBag only supports sum pooling"

        self._dummy_embedding_table_name = "dummy_embedding_table_name"
        self._dummy_feature_name = "dummy_feature_name"
        self.parameter_sharding: ParameterSharding = next(
            iter(table_name_to_parameter_sharding.values())
        )
        embedding_table_config = EmbeddingTableConfig(
            num_embeddings=module.num_embeddings,
            embedding_dim=module.embedding_dim,
            name=self._dummy_embedding_table_name,
            feature_names=[self._dummy_feature_name],
            pooling=PoolingType.SUM,
            # We set is_weighted to True for now,
            # if per_sample_weights is None in forward(),
            # we could assign a all-one vector to per_sample_weights
            is_weighted=True,
            embedding_names=[self._dummy_feature_name],
        )

        if self.parameter_sharding.sharding_type == ShardingType.TABLE_WISE.value:
            # TODO: enable it with correct semantics, see T104397332
            raise RuntimeError(
                "table-wise sharding on a single EmbeddingBag is not supported yet"
            )

        self._embedding_sharding: EmbeddingSharding[
            SparseFeatures, torch.Tensor
        ] = create_embedding_bag_sharding(
            sharding_type=self.parameter_sharding.sharding_type,
            embedding_configs=[
                (
                    embedding_table_config,
                    self.parameter_sharding,
                    next(iter(module.parameters())),
                ),
            ],
            env=env,
            device=device,
            permute_embeddings=True,
        )
        self._input_dist: nn.Module = self._embedding_sharding.create_input_dist()
        self._lookup: nn.Module = self._embedding_sharding.create_lookup(
            fused_params=fused_params
        )
        self._output_dist: nn.Module = self._embedding_sharding.create_output_dist()

        # Get all fused optimizers and combine them.
        optims = []
        for _, module in self._lookup.named_modules():
            if isinstance(module, FusedOptimizerModule):
                # modify param keys to match EmbeddingBag
                params: Mapping[str, Union[torch.Tensor, ShardedTensor]] = {}
                for param_key, weight in module.fused_optimizer.params.items():
                    # pyre-fixme[16]: `Mapping` has no attribute `__setitem__`.
                    params[param_key.split(".")[-1]] = weight
                module.fused_optimizer.params = params
                optims.append(("", module.fused_optimizer))
        self._optim: CombinedOptimizer = CombinedOptimizer(optims)

    # pyre-ignore [14]
    def input_dist(
        self,
        ctx: ShardedModuleContext,
        input: Tensor,
        offsets: Optional[Tensor] = None,
        per_sample_weights: Optional[Tensor] = None,
    ) -> Awaitable[SparseFeatures]:
        if per_sample_weights is None:
            per_sample_weights = torch.ones_like(input, dtype=torch.float)
        features = KeyedJaggedTensor(
            keys=[self._dummy_feature_name],
            values=input,
            offsets=offsets,
            weights=per_sample_weights,
        )
        return self._input_dist(
            SparseFeatures(id_list_features=None, id_score_list_features=features,)
        ).wait()

    def compute(
        self, ctx: ShardedModuleContext, dist_input: SparseFeatures
    ) -> torch.Tensor:
        return self._lookup(dist_input)

    def output_dist(
        self, ctx: ShardedModuleContext, output: torch.Tensor
    ) -> LazyAwaitable[torch.Tensor]:
        return EmbeddingAwaitable(awaitable=self._output_dist(output),)

    # pyre-fixme[14]: `state_dict` overrides method defined in `Module` inconsistently.
    def state_dict(
        self,
        destination: Optional[Dict[str, Any]] = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> Dict[str, Any]:
        if destination is None:
            destination = OrderedDict()
            # pyre-ignore [16]
            destination._metadata = OrderedDict()
        lookup_state_dict = self._lookup.state_dict(None, "", keep_vars)
        # update key to match embeddingBag state_dict key
        for key, item in lookup_state_dict.items():
            new_key = prefix + key.split(".")[-1]
            destination[new_key] = item
        return destination

    def named_modules(
        self,
        memo: Optional[Set[nn.Module]] = None,
        prefix: str = "",
        remove_duplicate: bool = True,
    ) -> Iterator[Tuple[str, nn.Module]]:
        yield from [(prefix, self)]

    def named_parameters(
        self, prefix: str = "", recurse: bool = True
    ) -> Iterator[Tuple[str, nn.Parameter]]:
        for name, parameter in self._lookup.named_parameters("", recurse):
            # update name to match embeddingBag parameter name
            yield append_prefix(prefix, name.split(".")[-1]), parameter

    def sharded_parameter_names(self, prefix: str = "") -> Iterator[str]:
        if self.parameter_sharding.sharding_type == ShardingType.DATA_PARALLEL.value:
            yield from []
        else:
            for name, _ in self._lookup.named_parameters(""):
                yield append_prefix(prefix, name.split(".")[-1])

    def named_buffers(
        self, prefix: str = "", recurse: bool = True
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        for name, buffer in self._lookup.named_buffers("", recurse):
            yield append_prefix(prefix, name.split(".")[-1]), buffer

    def load_state_dict(
        self, state_dict: "OrderedDict[str, torch.Tensor]", strict: bool = True,
    ) -> _IncompatibleKeys:
        missing_keys = []
        unexpected_keys = []
        # update key to match  embeddingBag state_dict key
        for key, value in state_dict.items():
            new_key = ".".join([self._dummy_embedding_table_name, key])
            state_dict[new_key] = value
            state_dict.pop(key)
        missing, unexpected = self._lookup.load_state_dict(state_dict, strict,)
        missing_keys.extend(missing)
        unexpected_keys.extend(unexpected)

        return _IncompatibleKeys(
            missing_keys=missing_keys, unexpected_keys=unexpected_keys
        )

    def sparse_grad_parameter_names(
        self, destination: Optional[List[str]] = None, prefix: str = "",
    ) -> List[str]:
        destination = [] if destination is None else destination
        # pyre-ignore [29]
        lookup_sparse_grad_parameter_names = self._lookup.sparse_grad_parameter_names(
            None, ""
        )
        for name in lookup_sparse_grad_parameter_names:
            destination.append(name.split(".")[-1])
        return destination

    @property
    def fused_optimizer(self) -> KeyedOptimizer:
        return self._optim


class EmbeddingBagSharder(BaseEmbeddingSharder[nn.EmbeddingBag]):
    """
    This implementation uses non-fused `nn.EmbeddingBag`
    """

    def shard(
        self,
        module: nn.EmbeddingBag,
        params: Dict[str, ParameterSharding],
        env: ShardingEnv,
        device: Optional[torch.device] = None,
    ) -> ShardedEmbeddingBag:
        return ShardedEmbeddingBag(module, params, env, self.fused_params, device)

    def shardable_parameters(self, module: nn.EmbeddingBag) -> Dict[str, nn.Parameter]:
        return {name: param for name, param in module.named_parameters()}

    @property
    def module_type(self) -> Type[nn.EmbeddingBag]:
        return nn.EmbeddingBag
