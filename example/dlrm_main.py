#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import itertools
import random
import numpy as np
import os
import sys
import time
from dataclasses import dataclass, field
from math import sqrt
from typing import cast, Iterator, List, Optional, Tuple
from copy import deepcopy

import torch
import torchmetrics as metrics
from pyre_extensions import none_throws
from torch import nn, distributed as dist
from torch.utils.data import DataLoader


from torchrec import EmbeddingBagCollection

from torchrec.datasets.utils import Batch
from torchrec.distributed import TrainPipelineSparseDist
from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder
from torchrec.distributed.embedding_types import (
    EmbeddingComputeKernel,
    UnifiedEmbeddingConfig,
)
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.distributed.types import ModuleSharder
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingCollection
from torchrec.optim.keyed import CombinedOptimizer, KeyedOptimizerWrapper
from tqdm import tqdm

# torch.profiler
import torch.profiler as profiler
from torch.profiler import record_function, tensorboard_trace_handler


# OSS import
try:
    # pyre-ignore[21]
    # @manual=//torchrec/github/examples/dlrm/data:dlrm_dataloader
    from data.dlrm_dataloader import get_dataloader, STAGES

    # pyre-ignore[21]
    # @manual=//torchrec/github/examples/dlrm/modules:dlrm_train
    from modules.dlrm_train import DLRMTrain, DeepFM
except ImportError:
    pass

# internal import
try:
    from .data.dlrm_dataloader import (  # noqa F811
        get_dataloader,
        STAGES,
    )
    from .modules.dlrm_train import DLRMTrain, DeepFM  # noqa F811
except ImportError:
    pass

TRAIN_PIPELINE_STAGES = 3  # Number of stages in TrainPipelineSparseDist.


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="torchrec dlrm example trainer")
    parser.add_argument(
        "--epochs", type=int, default=1, help="number of epochs to train"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="batch size to use for training"
    )
    parser.add_argument(
        "--limit_train_batches", type=int, default=15, help="number of train batches",
    )
    parser.add_argument(
        "--limit_val_batches",
        type=int,
        default=None,
        help="number of validation batches",
    )
    parser.add_argument(
        "--limit_test_batches", type=int, default=None, help="number of test batches",
    )
    parser.add_argument(
        "--num_workers", type=int, default=8, help="number of dataloader workers",
    )
    parser.add_argument(
        "--num_embeddings",
        type=int,
        default=100_000,
        help="max_ind_size. The number of embeddings in each embedding table. Defaults"
        " to 100_000 if num_embeddings_per_feature is not supplied.",
    )
    parser.add_argument(
        "--num_embeddings_per_feature",
        type=str,
        default=None,
        help="Comma separated max_ind_size per sparse feature. The number of embeddings"
        " in each embedding table. 26 values are expected for the Criteo dataset.",
    )
    parser.add_argument(
        "--dense_arch_layer_sizes",
        type=str,
        default="512,256,64",
        help="Comma separated layer sizes for dense arch.",
    )
    parser.add_argument(
        "--over_arch_layer_sizes",
        type=str,
        default="512,512,256,1",
        help="Comma separated layer sizes for over arch.",
    )
    parser.add_argument(
        "--embedding_dim", type=int, default=128, help="Size of each embedding.",
    )
    parser.add_argument(
        "--undersampling_rate",
        type=float,
        help="Desired proportion of zero-labeled samples to retain (i.e. undersampling zero-labeled rows)."
        " Ex. 0.3 indicates only 30pct of the rows with label 0 will be kept."
        " All rows with label 1 will be kept. Value should be between 0 and 1."
        " When not supplied, no undersampling occurs.",
    )
    parser.add_argument(
        "--lr_change_point",
        type=float,
        default=0.80,
        help="The point through training at which learning rate should change to the value set by"
        " lr_after_change_point. The default value is 0.80 which means that 80% through the total iterations (totaled"
        " across all epochs), the learning rate will change.",
    )
    parser.add_argument(
        "--lr_after_change_point",
        type=float,
        default=0.20,
        help="Learning rate after change point in first epoch.",
    )
    parser.add_argument(
        "--seed", type=float, help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--pin_memory",
        dest="pin_memory",
        action="store_true",
        help="Use pinned memory when loading data.",
    )
    parser.add_argument(
        "--in_memory_binary_criteo_path",
        type=str,
        default="./dataset/npy_format",
        # default=None,
        help="Path to a folder containing the binary (npy) files for the Criteo dataset."
        " When supplied, InMemoryBinaryCriteoIterDataPipe is used.",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=15.0, help="Learning rate.",
    )
    parser.add_argument(
        "--shuffle_batches",
        type=bool,
        default=False,
        help="Shuffle each batch during training.",
    )
    # unified flag
    parser.add_argument(
        "--use_unified", type=bool, default=False, help="Original or unified table",
    )

    parser.add_argument(
        "--hot_entries_path", type=str, default="./dataset/criteo_hot_entries_full.npy",
    )

    parser.add_argument(
        "--n_hot_entries", type=int, default=-1,
    )
    parser.add_argument(
        "--use_prefetch", type=bool, default=False, help="Use prefetch cold entries"
    )
    parser.add_argument(
        "--model", type=str, default="DLRM", help="Rec Model: [DLRM, WDL, DCN, DFM]"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="criteo",
        help="Rec dataset: [criteo, avazu, criteo-tb]",
    )
    parser.add_argument(
        "--uvm_ratio", type=float, default=0.0, help="cold entries uvm ratio"
    )

    parser.add_argument(
        "--cold_entries_sharding_type",
        type=str,
        default="rw",
        help="Currently rw supported",
    )
    parser.add_argument(
        "--system_name",
        type=str,
        default="TorchRec",
        help="DLRM system: [FEC, TorchRec, Parallax, FLECHE]",
    )
    parser.set_defaults(pin_memory=None)
    return parser.parse_args(argv)


def _evaluate(
    args: argparse.Namespace,
    train_pipeline: TrainPipelineSparseDist,
    iterator: Iterator[Batch],
    next_iterator: Iterator[Batch],
    stage: str,
    epoch: int,
) -> Tuple[float, float]:
    """
    Evaluate model. Computes and prints metrics including AUROC and Accuracy. Helper
    function for train_val_test.

    Args:
        args (argparse.Namespace): parsed command line args.
        train_pipeline (TrainPipelineSparseDist): pipelined model.
        iterator (Iterator[Batch]): Iterator used for val/test batches.
        next_iterator (Iterator[Batch]): Iterator used for the next phase (either train
            if there are more epochs to train on or test if all epochs are complete).
            Used to queue up the next TRAIN_PIPELINE_STAGES - 1 batches before
            train_val_test switches to the next phase. This is done so that when the
            next phase starts, the first output train_pipeline generates an output for
            is the 1st batch for that phase.
        stage (str): "val" or "test".

    Returns:
        None.
    """
    model = train_pipeline._model
    model.eval()
    device = train_pipeline._device
    limit_batches = (
        args.limit_val_batches if stage == "val" else args.limit_test_batches
    )
    if limit_batches is not None:
        limit_batches -= TRAIN_PIPELINE_STAGES - 1

    # Because TrainPipelineSparseDist buffer batches internally, we load in
    # TRAIN_PIPELINE_STAGES - 1 batches from the next_iterator into the buffers so that
    # when train_val_test switches to the next phase, train_pipeline will start
    # producing results for the TRAIN_PIPELINE_STAGES - 1 buffered batches (as opposed
    # to the last TRAIN_PIPELINE_STAGES - 1 batches from iterator).

    combined_iterator = itertools.chain(
        iterator
        if limit_batches is None
        else itertools.islice(iterator, limit_batches),
        itertools.islice(next_iterator, TRAIN_PIPELINE_STAGES - 1),
    )
    auroc = metrics.AUROC(compute_on_step=False).to(device)
    accuracy = metrics.Accuracy(compute_on_step=False).to(device)

    # Infinite iterator instead of while-loop to leverage tqdm progress bar.
    logits_all = []
    labels_all = []
    for _ in tqdm(iter(int, 1), desc=f"Evaluating {stage} set"):
        try:
            _loss, logits, labels = train_pipeline.progress(combined_iterator)
            auroc(logits, labels)
            accuracy(logits, labels)
        except StopIteration:
            break
    auroc_result = auroc.compute().item()
    accuracy_result = accuracy.compute().item()
    rank = dist.get_rank()
    if rank == 0:
        print(f"[Note]Rk#{rank}: AUROC over {stage} set: {auroc_result}.")


def _train(
    args: argparse.Namespace,
    train_pipeline: TrainPipelineSparseDist,
    iterator: Iterator[Batch],
    next_iterator: Optional[Iterator[Batch]] = None,
    epoch: int = 0,
) -> None:
    """
    Train model for 1 epoch. Helper function for train_val_test.

    Args:
        args (argparse.Namespace): parsed command line args.
        train_pipeline (TrainPipelineSparseDist): pipelined model.
        iterator (Iterator[Batch]): Iterator used for training batches.
        next_iterator (Iterator[Batch]): Iterator used for validation batches. Used to
            queue up the next TRAIN_PIPELINE_STAGES - 1 batches before train_val_test
            switches to validation mode. This is done so that when validation starts,
            the first output train_pipeline generates an output for is the 1st
            validation batch (as opposed to a buffered train batch).
        epoch (int): Which epoch the model is being trained on.

    Returns:
        None.
    """

    train_pipeline._model.train()

    limit_batches = args.limit_train_batches
    print(f"[Note]_train: limit_batches: {limit_batches}")
    # For the first epoch, train_pipeline has no buffered batches, but for all other
    # epochs, train_pipeline will have TRAIN_PIPELINE_STAGES - 1 from iterator already
    # present in its buffer.
    if limit_batches is not None and epoch > 0:
        limit_batches -= TRAIN_PIPELINE_STAGES - 1

    # Because TrainPipelineSparseDist buffer batches internally, we load in
    # TRAIN_PIPELINE_STAGES - 1 batches from the next_iterator into the buffers so that
    # when train_val_test switches to the next phase, train_pipeline will start
    # producing results for the TRAIN_PIPELINE_STAGES - 1 buffered batches (as opposed
    # to the last TRAIN_PIPELINE_STAGES - 1 batches from iterator).
    """
    combined_iterator = itertools.chain(
        iterator
        if args.limit_train_batches is None
        else itertools.islice(iterator, limit_batches),
        itertools.islice(next_iterator, TRAIN_PIPELINE_STAGES - 1),
    )
    """

    train_iterator = itertools.islice(iterator, limit_batches)
    '''
    activities = [profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA]
    schedule = torch.profiler.schedule(
        skip_first=0, wait=1, warmup=2, active=10, repeat=1
    )
    system_name = args.system_name
    if args.use_prefetch and system_name == "FEC":
        system_name += "_prefetch"

    world_size = dist.get_world_size()
    profiler_log_path = f"./{args.dataset_name}/{system_name}_{args.model}_w{world_size}_bs{args.batch_size}_hot{args.n_hot_entries}_dim{args.embedding_dim}"
    print(f"[Note]Save to {profiler_log_path}")

    with profiler.profile(
        activities=activities,
        schedule=schedule,
        on_trace_ready=tensorboard_trace_handler(profiler_log_path),
    ) as prof:
    '''
    if True:
        pbar = tqdm(iter(int, 1), desc=f"Epoch {epoch}")
        for it in itertools.count():
            try:
                train_pipeline.progress(train_iterator)
                #prof.step()

                if dist.get_rank() == 0:
                    pbar.update(1)

            except StopIteration:
                break


@dataclass
class TrainValTestResults:
    val_accuracies: List[float] = field(default_factory=list)
    val_aurocs: List[float] = field(default_factory=list)
    test_accuracy: Optional[float] = None
    test_auroc: Optional[float] = None


def train_val_test(
    args: argparse.Namespace,
    train_pipeline: TrainPipelineSparseDist,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    test_dataloader: DataLoader,
) -> TrainValTestResults:
    """
    Train/validation/test loop. Contains customized logic to ensure each dataloader's
    batches are used for the correct designated purpose (train, val, test). This logic
    is necessary because TrainPipelineSparseDist buffers batches internally (so we
    avoid batches designated for one purpose like training getting buffered and used for
    another purpose like validation).

    Args:
        args (argparse.Namespace): parsed command line args.
        train_pipeline (TrainPipelineSparseDist): pipelined model.
        train_dataloader (DataLoader): DataLoader used for training.
        val_dataloader (DataLoader): DataLoader used for validation.
        test_dataloader (DataLoader): DataLoader used for testing.

    Returns:
        TrainValTestResults.
    """

    train_val_test_results = TrainValTestResults()

    train_iterator = iter(train_dataloader)
    test_iterator = iter(test_dataloader)
    for epoch in range(args.epochs):
        val_iterator = iter(val_dataloader)

        _train(args, train_pipeline, train_iterator, val_iterator, epoch)

        train_iterator = iter(train_dataloader)

        val_next_iterator = (
            test_iterator if epoch == args.epochs - 1 else train_iterator
        )
        val_next_iterator = train_iterator
        _evaluate(
            args, train_pipeline, val_iterator, val_next_iterator, "val", epoch,
        )

    test_accuracy, test_auroc = _evaluate(
        args, train_pipeline, test_iterator, iter(test_dataloader), "test", 2333,
    )
    train_val_test_results.test_accuracy = test_accuracy
    train_val_test_results.test_auroc = test_auroc
    return train_val_test_results


def build_unified_cw_and_dp_config(
    args: argparse.Namespace,
) -> Tuple[UnifiedEmbeddingConfig, UnifiedEmbeddingConfig]:
    cw_compute_kernel: EmbeddingComputeKernel = EmbeddingComputeKernel.BATCHED_FUSED
    dp_compute_kernel: EmbeddingComputeKernel = EmbeddingComputeKernel.BATCHED_DENSE
    world_size: int = dist.get_world_size()
    rank: int = dist.get_rank()
    print(f"[Note]Build hot&cold config of rk#{rank} / {world_size}")

    n_hot_entries: int = args.n_hot_entries
    hot_entries_path: str = args.hot_entries_path
    hot_entries_np_arr: np.array = np.load(hot_entries_path)
    print(
        f"[Note]build_unified_cw_and_dp_config, from {hot_entries_path}, use {n_hot_entries} / {hot_entries_np_arr.shape}"
    )
    hash_sizes: List[int] = args.num_embeddings_per_feature
    n_tables: int = len(hash_sizes)

    n_table_hot_entries: List[int] = [0] * n_tables

    # (n_hot_entries, dim1) = hot_entries_np_arr.shape

    for hot_entry_index in range(n_hot_entries):
        table_idx: int = hot_entries_np_arr[hot_entry_index][0]
        entry_idx: int = hot_entries_np_arr[hot_entry_index][1]
        n_table_hot_entries[table_idx] += 1

    # NOTE Build hot-dp & cold-cw unified config
    hot_embedding_tables_rows: List[int] = []
    cold_embedding_tables_rows: List[int] = []

    hot_weight_init_mins: List[float] = []
    hot_weight_init_maxs: List[float] = []

    cold_weight_init_mins: List[float] = []
    cold_weight_init_maxs: List[float] = []

    for table_idx in range(n_tables):
        # assert n_table_hot_entries[table_idx] > 0
        # table contains hot idx
        if n_table_hot_entries[table_idx] > 0:
            hot_embedding_tables_rows.append(n_table_hot_entries[table_idx])
            hot_weight_init_mins.append(-sqrt(1 / hash_sizes[table_idx]))
            hot_weight_init_maxs.append(sqrt(1 / hash_sizes[table_idx]))
        # table contains cold idx
        if hash_sizes[table_idx] - n_table_hot_entries[table_idx] > 0:
            cold_embedding_tables_rows.append(
                hash_sizes[table_idx] - n_table_hot_entries[table_idx]
            )
            cold_weight_init_mins.append(-sqrt(1 / hash_sizes[table_idx]))
            cold_weight_init_maxs.append(sqrt(1 / hash_sizes[table_idx]))

    this_worker_cold_embedding_tables_rows: List[int] = []
    this_worker_cold_weight_init_mins: List[float] = []
    this_worker_cold_weight_init_maxs: List[float] = []
    rw_offsets = None
    if args.cold_entries_sharding_type == "rw":
        # First, get each worker's #(hash_size)
        total_cold_hash_size: int = sum(cold_embedding_tables_rows)
        fir_cold_hash_size_per_worker: int = total_cold_hash_size // world_size
        lef_cold_hash_size: int = total_cold_hash_size - fir_cold_hash_size_per_worker * world_size
        cold_hash_size_per_worker: List[int] = [
            fir_cold_hash_size_per_worker
        ] * world_size
        for i in range(lef_cold_hash_size):
            cold_hash_size_per_worker[i] += 1
        cold_hash_offsets_per_worker: List[int] = list(
            itertools.accumulate(cold_hash_size_per_worker)
        )
        rw_offsets = (
            cold_hash_offsets_per_worker[rank] - cold_hash_size_per_worker[rank]
        )
        # Second, get the range of worker's hash_size
        this_worker_en: int = cold_hash_offsets_per_worker[rank]
        this_worker_st: int = this_worker_en - cold_hash_size_per_worker[rank]
        pre_sum: int = 0

        for i in range(len(cold_embedding_tables_rows)):
            # get range
            lb = pre_sum
            rb = pre_sum + cold_embedding_tables_rows[i]
            pre_sum += cold_embedding_tables_rows[i]
            if rb <= this_worker_st or lb >= this_worker_en:
                continue
            else:
                i_st: int = max(lb, this_worker_st)
                i_en: int = min(rb, this_worker_en)
                this_worker_cold_embedding_tables_rows.append(i_en - i_st)
                this_worker_cold_weight_init_mins.append(cold_weight_init_mins[i])
                this_worker_cold_weight_init_maxs.append(cold_weight_init_maxs[i])
    else:
        assert False

    # Last, generate
    hot_embedding_tables_offsets: List[int] = [0] + list(
        itertools.accumulate(hot_embedding_tables_rows)
    )
    cold_embedding_tables_offsets: List[int] = [0] + list(
        itertools.accumulate(this_worker_cold_embedding_tables_rows)
    )

    hot_unified_table_row = hot_embedding_tables_offsets[-1]
    cold_unified_table_row = cold_embedding_tables_offsets[-1]

    # check
    assert cold_unified_table_row == sum(this_worker_cold_embedding_tables_rows)

    hot_unified_table_col: int = args.embedding_dim
    cold_unified_table_col: int = args.embedding_dim

    unified_cw_config: UnifiedEmbeddingConfig = UnifiedEmbeddingConfig(
        compute_kernel=cw_compute_kernel,
        embedding_tables_rows=this_worker_cold_embedding_tables_rows,
        embedding_table_offsets=cold_embedding_tables_offsets,
        weight_init_mins=this_worker_cold_weight_init_mins,
        weight_init_maxs=this_worker_cold_weight_init_maxs,
        unified_table_row=cold_unified_table_row,
        unified_table_col=cold_unified_table_col,
        sharding_type=args.cold_entries_sharding_type,
        rw_offsets=rw_offsets,
        uvm_ratio=args.uvm_ratio,
    )

    unified_dp_config: UnifiedEmbeddingConfig = UnifiedEmbeddingConfig(
        compute_kernel=dp_compute_kernel,
        embedding_tables_rows=hot_embedding_tables_rows,
        embedding_table_offsets=hot_embedding_tables_offsets,
        weight_init_mins=hot_weight_init_mins,
        weight_init_maxs=hot_weight_init_maxs,
        unified_table_row=hot_unified_table_row,
        unified_table_col=hot_unified_table_col,
        sharding_type="dp",
    )
    return (unified_cw_config, unified_dp_config)


def main(argv: List[str]) -> None:
    """
    Trains, validates, and tests a Deep Learning Recommendation Model (DLRM)
    (https://arxiv.org/abs/1906.00091). The DLRM model contains both data parallel
    components (e.g. multi-layer perceptrons & interaction arch) and model parallel
    components (e.g. embedding tables). The DLRM model is pipelined so that dataloading,
    data-parallel to model-parallel comms, and forward/backward are overlapped. Can be
    run with either a random dataloader or an in-memory Criteo 1 TB click logs dataset
    (https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/).

    Args:
        argv (List[str]): command line args.

    Returns:
        None.
    """

    args = parse_args(argv)
    print(f"args:{args}")
    # check args
    # check system

    assert args.system_name in ["FEC", "TorchRec", "Parallax", "FLECHE"]
    if args.use_unified:
        if args.system_name != "FEC":
            print(
                f"[Note]sys:{args.system_name} but with flag use_unified {args.use_unified}"
            )
            args.use_unified = False
    else:
        if args.system_name == "FEC":
            print(
                f"[Note]sys:{args.system_name} but with flag use_unified {args.use_unified}"
            )
            args.use_unified = True

    assert (args.use_unified and args.system_name == "FEC") or (
        not args.use_unified and args.system_name != "FEC"
    )

    rank = int(os.environ["LOCAL_RANK"])

    if torch.cuda.is_available():
        device: torch.device = torch.device(f"cuda:{rank}")
        backend = "nccl"
        torch.cuda.set_device(device)
    else:
        device: torch.device = torch.device("cpu")
        backend = "gloo"

    if not torch.distributed.is_initialized():
        dist.init_process_group(backend=backend)

    if args.use_unified and args.n_hot_entries == -1:
        if args.dataset_name == "criteo":
            # criteo
            esti_n_hot_entries = [7225, 12641, 21079, 33578, 52664, 82069]
        elif args.dataset_name == "avazu":
            # avazu
            esti_n_hot_entries = [1864, 2795, 4076, 5971, 9384, 18587]
        elif args.dataset_name == "criteo-tb":
            # criteo-tb
            esti_n_hot_entries = [6435, 12000, 21681, 36803, 58865, 92021]
        else:
            raise NotImplementedError(f"Dataset {args.dataset_name} not supported yet!")
        args.n_hot_entries = esti_n_hot_entries[
            int(np.log2(dist.get_world_size()))
            - 1
            + int(np.log2(args.batch_size // 1024))
            - 2
        ]
        print(f"[Note]Use estimated Optimal #hot:{args.n_hot_entries}")

    if args.num_embeddings_per_feature is not None:
        args.num_embeddings_per_feature = list(
            map(int, args.num_embeddings_per_feature.split(","))
        )
        args.num_embeddings = None

    train_dataloader = get_dataloader(args, backend, "train")

    from torchrec.datasets.criteo import DEFAULT_CAT_NAMES, DEFAULT_INT_NAMES

    eb_configs = [
        EmbeddingBagConfig(
            name=f"t_{feature_name}",
            embedding_dim=args.embedding_dim,
            num_embeddings=none_throws(args.num_embeddings_per_feature)[feature_idx]
            if args.num_embeddings is None
            else args.num_embeddings,
            feature_names=[feature_name],
        )
        for feature_idx, feature_name in enumerate(DEFAULT_CAT_NAMES)
    ]
    sharded_module_kwargs = {}
    if args.over_arch_layer_sizes is not None:
        sharded_module_kwargs["over_arch_layer_sizes"] = list(
            map(int, args.over_arch_layer_sizes.split(","))
        )
    assert args.model in [
        "DLRM",
        "WDL",
        "DFM",
        "DCN",
    ], f"Model{args.model} not supported!"

    if args.model == "DLRM":
        train_model = DLRMTrain(
            embedding_bag_collection=EmbeddingBagCollection(
                tables=eb_configs, device=torch.device("meta")
            ),
            dense_in_features=len(DEFAULT_INT_NAMES),
            dense_arch_layer_sizes=list(
                map(int, args.dense_arch_layer_sizes.split(","))
            ),
            over_arch_layer_sizes=list(map(int, args.over_arch_layer_sizes.split(","))),
            dense_device=device,
            use_unified=args.use_unified,
        )
    elif args.model in ["WDL", "DFM", "DCN"]:
        train_model = DeepFM(
            embedding_bag_collection=EmbeddingBagCollection(
                tables=eb_configs, device=torch.device("meta")
            ),
            dense_in_features=len(DEFAULT_INT_NAMES),
            hidden_layer_size=20,
            deep_fm_dimension=5,
            model_name=args.model,
            use_unified=args.use_unified,
        )
    else:
        raise NotImplementedError(f"Model{args.model} not implemented yet!")

    fused_params = {
        "learning_rate": args.learning_rate,
    }
    sharders = [
        EmbeddingBagCollectionSharder(fused_params=fused_params),
    ]

    unified_cw_config = None
    unified_dp_config = None
    if args.use_unified:
        unified_cw_config, unified_dp_config = build_unified_cw_and_dp_config(
            args=args,
        )

    model = DistributedModelParallel(
        module=train_model,
        device=device,
        batch_size=args.batch_size,
        sharders=cast(List[ModuleSharder[nn.Module]], sharders),
        system_name=args.system_name,
        dataset_name=args.dataset_name,
        use_unified=args.use_unified,
        use_prefetch=args.use_prefetch,
        unified_cw_config=unified_cw_config,
        unified_dp_config=unified_dp_config,
    )
    print(f"[Note][DLRM Train] Done build DistributedModelParallel")

    dense_optimizer = KeyedOptimizerWrapper(
        dict(model.named_parameters()),
        lambda params: torch.optim.SGD(params, lr=args.learning_rate),
    )

    optimizer = CombinedOptimizer([model.fused_optimizer, dense_optimizer])
    train_pipeline = TrainPipelineSparseDist(
        model, optimizer, device, rank, args.use_unified
    )

    train_iterator = iter(train_dataloader)
    _train(args, train_pipeline, train_iterator)


if __name__ == "__main__":
    main(sys.argv[1:])
