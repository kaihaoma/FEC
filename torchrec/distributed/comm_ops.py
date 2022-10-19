#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, TypeVar, Any

import torch
import torch.distributed as dist
from torch import Tensor
from torch.autograd import Function
from torch.autograd.profiler import record_function
from torchrec.distributed.types import Awaitable, NoWait

try:
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops")
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_cpu")
except OSError:
    pass


# OSS
try:
    import fbgemm_gpu  # @manual # noqa
except ImportError:
    pass


W = TypeVar("W")

# TODO: T96382816, NE Parity Backward compatibility
GRADIENT_DIVISION: bool = True


def set_gradient_division(val: bool) -> None:
    global GRADIENT_DIVISION
    GRADIENT_DIVISION = val


"""
Some commonly used notations for comm ops:
    B - batch size
    T - number of embedding tables
    D - embedding dimension
"""


class Request(Awaitable[W]):
    """
    Defines a collective operation request for a process group on a tensor.

    Args:
        pg (dist.ProcessGroup): The process group the request is for.
    """

    # pyre-fixme[11]: Annotation `ProcessGroup` is not defined as a type.
    def __init__(self, pg: dist.ProcessGroup) -> None:
        super().__init__()
        self.pg: dist.ProcessGroup = pg
        # pyre-fixme[11]: Annotation dist.Work is not defined as a type.
        self.req: Optional[dist.Work] = None
        self.tensor: Optional[W] = None
        self.a2ai = None  # type: ignore
        self.rsi = None  # type: ignore
        self.wait_function = None  # type: ignore

    def _wait_impl(self) -> W:
        """
        Calls the wait function for this request.
        """

        ret = self.wait_function.apply(self.pg, self, self.tensor)
        self.req = None
        self.tensor = None
        return ret


@dataclass
class RWAll2AllPooledInfo(object):
    # use for unified-rw

    emb_dim: int
    input_batch_size_per_rank: List[int]
    output_batch_size_per_rank: List[int]


@dataclass
class All2AllPooledInfo(object):
    """
    The data class that collects the attributes when calling the `alltoall_pooled`
    operation.

    Attributes:
        batch_size_per_rank (List[int]): batch size in each rank
        dim_sum_per_rank (List[int]): number of features (sum of dimensions) of the
            embedding in each rank.
        dim_sum_per_rank_tensor (Optional[Tensor]): the tensor version of
            `dim_sum_per_rank`, this is only used by the fast kernel of
            `_recat_pooled_embedding_grad_out`.
        cumsum_dim_sum_per_rank_tensor (Optional[Tensor]): cumulative sum of
            `dim_sum_per_rank`, this is only used by the fast kernel of
            `_recat_pooled_embedding_grad_out`.
        B_local (int): local batch size before scattering.
    """

    batch_size_per_rank: List[int]
    dim_sum_per_rank: List[int]
    dim_sum_per_rank_tensor: Optional[Tensor]
    cumsum_dim_sum_per_rank_tensor: Optional[Tensor]


@dataclass
class All2AllSequenceInfo(object):
    """
    The data class that collects the attributes when calling the `alltoall_sequence`
    operation.

    Attributes:
        embedding_dim (int): embedding dimension.
        lengths_after_sparse_data_all2all (Tensor): lengths of sparse features after
            AlltoAll.
        forward_recat_tensor (Tensor): recat tensor for forward.
        backward_recat_tensor (Tensor): recat tensor for backward.
        input_splits (List[int]): input splits.
        output_splits (List[int]): output splits.
        lengths_sparse_before_features_all2all (Optional[Tensor]): lengths of sparse
            features before AlltoAll.
    """

    embedding_dim: int
    lengths_after_sparse_data_all2all: Tensor
    forward_recat_tensor: Tensor
    backward_recat_tensor: Tensor
    input_splits: List[int]
    output_splits: List[int]
    permuted_lengths_after_sparse_data_all2all: Optional[Tensor] = None


@dataclass
class All2AllVInfo(object):
    """
    The data class that collects the attributes when calling the `alltoallv` operation.

    Attributes:
        dim_sum_per_rank (List[int]): number of features (sum of dimensions) of the
            embedding in each rank.
        B_global (int): global batch size for each rank.
        B_local (int): local batch size before scattering.
        B_local_list: (List[int]): local batch sizes for each embedding table locally
            (in my current rank).
        D_local_list (List[int]): embedding dimension of each embedding table locally
            (in my current rank).
        input_split_sizes (List[int]): The input split sizes for each rank, this
            remembers how to split the input when doing the `all_to_all_single` operation.
        output_split_sizes (List[int]): The output split sizes for each rank, this
            remembers how to fill the output when doing the `all_to_all_single` operation.
    """

    dims_sum_per_rank: List[int]
    B_global: int
    B_local: int
    B_local_list: List[int]
    D_local_list: List[int]
    input_split_sizes: List[int] = field(default_factory=list)
    output_split_sizes: List[int] = field(default_factory=list)


@dataclass
class ReduceScatterInfo(object):
    """
    The data class that collects the attributes when calling the `reduce_scatter_pooled`
    operation.

    Attributes:
        input_sizes (List[int]): the sizes of the input tensors. This remembers the
            sizes of the input tensors when running the backward pass and producing the
            gradient.
    """

    input_sizes: List[int]


@dataclass
class All2AllDenseInfo(object):
    """
    The data class that collects the attributes when calling the alltoall_dense
    operation.
    """

    output_splits: List[int]
    batch_size: int
    input_shape: List[int]
    input_splits: List[int]


def _get_split_lengths_by_len(
    world_size: int, my_rank: int, n: int
) -> Tuple[int, List[int]]:
    k, m = divmod(n, world_size)
    if m == 0:
        splits = [k] * world_size
        my_len = k
    else:
        splits = [(k + 1) if i < m else k for i in range(world_size)]
        my_len = splits[my_rank]
    return (my_len, splits)


def alltoall_pooled(
    a2a_pooled_embs_tensor: Tensor,
    batch_size_per_rank: List[int],
    dim_sum_per_rank: List[int],
    dim_sum_per_rank_tensor: Optional[Tensor] = None,
    cumsum_dim_sum_per_rank_tensor: Optional[Tensor] = None,
    group: Optional[dist.ProcessGroup] = None,
) -> Awaitable[Tensor]:
    """
    Performs AlltoAll operation for a single pooled embedding tensor. Each process
    splits the input pooled embeddings tensor based on the world size, and then scatters
    the split list to all processes in the group. Then concatenates the received tensors
    from all processes in the group and returns a single output tensor.

    Args:
        a2a_pooled_embs_tensor (Tensor): input pooled embeddings. Must be pooled
            together before passing into this function. Its shape is B x D_local_sum,
            where D_local_sum is the dimension sum of all the local
            embedding tables.
        batch_size_per_rank (List[int]): batch size in each rank.
        dim_sum_per_rank (List[int]): number of features (sum of dimensions) of the
            embedding in each rank.
        dim_sum_per_rank_tensor (Optional[Tensor]): the tensor version of
            `dim_sum_per_rank`, this is only used by the fast kernel of
            `_recat_pooled_embedding_grad_out`.
        cumsum_dim_sum_per_rank_tensor (Optional[Tensor]): cumulative sum of
            `dim_sum_per_rank`, this is only used by the fast kernel of
            `_recat_pooled_embedding_grad_out`.
        group (Optional[dist.ProcessGroup]): The process group to work on. If None, the
            default process group will be used.

    Returns:
        Awaitable[List[Tensor]]: async work handle (`Awaitable`), which can be `wait()` later to get the resulting tensor.

    .. warning::
        `alltoall_pooled` is experimental and subject to change.
    """

    if group is None:
        group = dist.distributed_c10d._get_default_group()

    if dist.get_world_size(group) <= 1:
        return NoWait(a2a_pooled_embs_tensor)

    myreq = Request(group)
    a2ai = All2AllPooledInfo(
        batch_size_per_rank=batch_size_per_rank,
        dim_sum_per_rank=dim_sum_per_rank,
        dim_sum_per_rank_tensor=dim_sum_per_rank_tensor,
        cumsum_dim_sum_per_rank_tensor=cumsum_dim_sum_per_rank_tensor,
    )
    # pyre-fixme[16]: `All2All_Pooled_Req` has no attribute `apply`.
    All2All_Pooled_Req.apply(group, myreq, a2ai, a2a_pooled_embs_tensor)
    return myreq


def rwalltoall_pooled(
    a2a_pooled_embs_tensor: Tensor,
    input_batch_size_per_rank: List[int],
    output_batch_size_per_rank: List[int],
    emb_dim: int,
    group: Optional[dist.ProcessGroup] = None,
) -> Awaitable[Tensor]:

    if group is None:
        group = dist.distributed_c10d._get_default_group()

    if dist.get_world_size(group) <= 1:
        return NoWait(a2a_pooled_embs_tensor)

    myreq = Request(group)
    a2ai = RWAll2AllPooledInfo(
        emb_dim=emb_dim,
        input_batch_size_per_rank=input_batch_size_per_rank,
        output_batch_size_per_rank=output_batch_size_per_rank,
    )
    # pyre-fixme[16]: `All2All_Pooled_Req` has no attribute `apply`.
    # All2All_Pooled_Req.apply(group, myreq, a2ai, a2a_pooled_embs_tensor)
    RWAll2All_Pooled_Req.apply(group, myreq, a2ai, a2a_pooled_embs_tensor)
    return myreq


def alltoall_sequence(
    # (T, B, L_i * D) flattened
    a2a_sequence_embs_tensor: Tensor,
    forward_recat_tensor: Tensor,
    backward_recat_tensor: Tensor,
    lengths_after_sparse_data_all2all: Tensor,
    input_splits: List[int],
    output_splits: List[int],
    group: Optional[dist.ProcessGroup] = None,
) -> Awaitable[Tensor]:
    """
    Performs AlltoAll operation for sequence embeddings. Each process splits the input
    tensor based on the world size, and then scatters the split list to all processes in
    the group. Then concatenates the received tensors from all processes in the group
    and returns a single output tensor.

    NOTE:
        AlltoAll operator for (T * B * L_i, D) tensors.
        Does not support mixed dimensions.

    Args:
        a2a_sequence_embs_tensor (Tensor): input embeddings. Usually with the shape of
            (T * B * L_i, D), where B - batch size, T - number of embedding tables,
            D - embedding dimension.
        forward_recat_tensor (Tensor): recat tensor for forward.
        backward_recat_tensor (Tensor): recat tensor for backward.
        lengths_after_sparse_data_all2all (Tensor): lengths of sparse features after
            AlltoAll.
        input_splits (Tensor): input splits.
        output_splits (Tensor): output splits.
        group (Optional[dist.ProcessGroup]): The process group to work on. If None, the
            default process group will be used.

    Returns:
        Awaitable[List[Tensor]]: async work handle (`Awaitable`), which can be `wait()` later to get the resulting tensor.

    .. warning::
        `alltoall_sequence` is experimental and subject to change.
    """

    if group is None:
        group = dist.distributed_c10d._get_default_group()

    if dist.get_world_size(group) <= 1:
        return NoWait(a2a_sequence_embs_tensor)

    myreq = Request(group)
    a2ai = All2AllSequenceInfo(
        embedding_dim=a2a_sequence_embs_tensor.shape[1],
        lengths_after_sparse_data_all2all=lengths_after_sparse_data_all2all,
        forward_recat_tensor=forward_recat_tensor,
        backward_recat_tensor=backward_recat_tensor,
        input_splits=input_splits,
        output_splits=output_splits,
    )
    # sequence of embeddings, bags are definitely non-uniform

    # pyre-fixme[16]: `All2All_Seq_Req` has no attribute `apply`.
    All2All_Seq_Req.apply(group, myreq, a2ai, a2a_sequence_embs_tensor)
    return myreq


def alltoallv(
    inputs: List[Tensor],
    out_split: Optional[List[int]] = None,
    per_rank_split_lengths: Optional[List[int]] = None,
    group: Optional[dist.ProcessGroup] = None,
) -> Awaitable[List[Tensor]]:
    """
    Performs `alltoallv` operation for a list of input embeddings. Each process scatters
    the list to all processes in the group.

    Args:
        input (List[Tensor]): list of tensors to scatter, one per rank. The tensors in
            the list usually have different lengths.
        out_split (Optional[List[int]]): output split sizes (or dim_sum_per_rank), if
            not specified, we will use `per_rank_split_lengths` to construct a output
            split with the assumption that all the embs have the same dimension.
        per_rank_split_lengths (Optional[List[int]]): split lengths per rank. If not
            specified, the `out_split` must be specified.
        group (Optional[dist.ProcessGroup]): The process group to work on. If None, the
            default process group will be used.

    Returns:
        Awaitable[List[Tensor]]: async work handle (`Awaitable`), which can be `wait()` later to get the resulting list of tensors.

    .. warning::
        `alltoallv` is experimental and subject to change.
    """

    if group is None:
        group = dist.distributed_c10d._get_default_group()

    world_size = dist.get_world_size(group)
    my_rank = dist.get_rank(group)

    myreq = Request(group)
    B_global, _ = inputs[0].size()
    D_local_list = [e.size()[1] for e in inputs]
    B_local, B_local_list = _get_split_lengths_by_len(world_size, my_rank, B_global)

    if out_split is not None:
        dims_sum_per_rank = out_split
    elif per_rank_split_lengths is not None:
        # all the embs have the same dimension
        dims_sum_per_rank = [s * D_local_list[0] for s in per_rank_split_lengths]
    else:
        raise RuntimeError("Need to specify either out_split or per_rank_split_lengths")

    a2ai = All2AllVInfo(
        dims_sum_per_rank=dims_sum_per_rank,
        B_local=B_local,
        B_local_list=B_local_list,
        D_local_list=D_local_list,
        B_global=B_global,
    )

    # pyre-fixme[16]: `All2Allv_Req` has no attribute `apply`.
    All2Allv_Req.apply(group, myreq, a2ai, inputs)

    return myreq


def reduce_scatter_pooled(
    inputs: List[Tensor], group: Optional[dist.ProcessGroup] = None,
) -> Awaitable[Tensor]:
    """
    Performs reduce-scatter operation for a pooled embeddings tensor split into world
    size number of chunks. The result of the reduce operation gets scattered to all
    processes in the group. Then concatenates the received tensors from all processes in
    the group and returns a single output tensor.

    Args:
        inputs (List[Tensor]): list of tensors to scatter, one per rank.
        group (Optional[dist.ProcessGroup]): The process group to work on. If None, the
            default process group will be used.

    Returns:
        Awaitable[List[Tensor]]: async work handle (Awaitable), which can be `wait()` later to get the resulting tensor.

    .. warning::
        `reduce_scatter_pooled` is experimental and subject to change.
    """

    if group is None:
        group = dist.distributed_c10d._get_default_group()

    if dist.get_world_size(group) <= 1:
        return NoWait(inputs[dist.get_rank(group)])

    myreq = Request(group)
    rsi = ReduceScatterInfo(input_sizes=[tensor.size() for tensor in inputs])
    # pyre-fixme[16]: `ReduceScatter_Req` has no attribute `apply`.
    ReduceScatter_Req.apply(group, myreq, rsi, *inputs)
    return myreq


# TODO: improve performance of _recat_pooled_embedding_grad_out, see T87591139
def _recat_pooled_embedding_grad_out(
    grad_output: Tensor, num_features_per_rank: List[int]
) -> Tensor:
    grad_outputs_by_rank = grad_output.split(num_features_per_rank, dim=1)
    return torch.cat(
        [
            grad_output_by_rank.contiguous().view(-1)
            for grad_output_by_rank in grad_outputs_by_rank
        ],
        dim=0,
    )


def _recat_seq_embedding(
    input_embeddings: Tensor,
    split_sizes: List[int],
    T_local: int,
    my_size: int,
    forward: bool,
) -> Tensor:
    seq_embeddings_by_rank = input_embeddings.split(split_sizes)
    if forward:
        return torch.cat(
            [
                seq_embeddings_by_rank[t * my_size + i]
                # .contiguous().view(-1)
                for i in range(my_size)
                for t in range(T_local)
            ],
            dim=0,
        )
    else:
        return torch.cat(
            [
                seq_embeddings_by_rank[i * T_local + t]
                # .contiguous()
                # .view(-1)
                for t in range(T_local)
                for i in range(my_size)
            ],
            dim=0,
        )


"""
class RWAll2AllPooledInfo(object):
    # use for unified-rw

    emb_dim: int
    input_batch_size_per_rank: List[int]
    output_batch_size_per_rank: List[int]
"""


class RWAll2All_Pooled_Req(Function):
    @staticmethod
    # pyre-fixme[14]: `forward` overrides method defined in `Function` inconsistently.
    def forward(
        # pyre-fixme[2]: Parameter must be annotated.
        ctx,
        pg: dist.ProcessGroup,
        myreq: Request[Tensor],
        a2ai: RWAll2AllPooledInfo,
        output_embeddings: Tensor,
    ) -> Tensor:

        my_rank = dist.get_rank(pg)
        (output_batch_size_sum, emb_dim) = output_embeddings.shape

        input_batch_size_per_rank = a2ai.input_batch_size_per_rank
        output_batch_size_per_rank = a2ai.output_batch_size_per_rank

        assert output_batch_size_sum == sum(
            output_batch_size_per_rank
        ), f"{output_batch_size_sum} != {sum(output_batch_size_per_rank)}"
        assert (
            emb_dim == a2ai.emb_dim
        ), f"emb_dim:{emb_dim} != a2ai.emb_dim:{a2ai.emb_dim}"

        sharded_input_embeddings = output_embeddings.view(-1)

        sharded_output_embeddings = torch.empty(
            sum(input_batch_size_per_rank) * emb_dim,
            dtype=output_embeddings.dtype,
            device=output_embeddings.device,
        )
        input_split_sizes = [bs * emb_dim for bs in output_batch_size_per_rank]
        output_split_sizes = [bs * emb_dim for bs in input_batch_size_per_rank]

        with record_function("## alltoall_fwd_single ##"):
            req = dist.all_to_all_single(
                output=sharded_output_embeddings,
                input=sharded_input_embeddings,
                output_split_sizes=output_split_sizes,
                input_split_sizes=input_split_sizes,
                group=pg,
                async_op=True,
            )

        myreq.req = req
        myreq.tensor = sharded_output_embeddings
        myreq.a2ai = a2ai
        myreq.wait_function = RWAll2All_Pooled_Wait
        ctx.myreq = myreq
        ctx.pg = pg
        return sharded_output_embeddings

    @staticmethod
    # pyre-fixme[2]: Parameter must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def backward(ctx, *unused) -> Tuple[None, None, None, Tensor]:

        pg = ctx.pg
        my_rank = dist.get_rank(pg)
        myreq = ctx.myreq
        a2ai = myreq.a2ai
        myreq.req.wait()
        myreq.req = None
        grad_output = myreq.tensor
        emb_dim = a2ai.emb_dim
        output_batch_size_per_rank = a2ai.output_batch_size_per_rank
        output_batch_size_sum = sum(output_batch_size_per_rank)
        assert output_batch_size_sum * emb_dim == grad_output.numel()
        grad_input = grad_output.view(output_batch_size_sum, emb_dim)
        if GRADIENT_DIVISION:
            grad_input.div_(dist.get_world_size(ctx.pg))
        myreq.tensor = None

        return (None, None, None, grad_input)


class RWAll2All_Pooled_Wait(Function):
    @staticmethod
    # pyre-fixme[14]: `forward` overrides method defined in `Function` inconsistently.
    def forward(
        # pyre-fixme[2]: Parameter must be annotated.
        ctx,
        pg: dist.ProcessGroup,
        myreq: Request[Tensor],
        sharded_output_embeddings: Tensor,
    ) -> Tensor:
        my_rank = dist.get_rank(pg)
        a2ai = myreq.a2ai
        ctx.a2ai = a2ai
        myreq.req.wait()
        myreq.req = None
        myreq.tensor = None
        ctx.pg = pg
        ctx.myreq = myreq
        emb_dim = a2ai.emb_dim
        input_batch_size_per_rank = a2ai.input_batch_size_per_rank
        # output_batch_size_per_rank = a2ai.output_batch_size_per_rank

        result = sharded_output_embeddings.reshape(-1, emb_dim)
        # result must be a Tensor with #[Entry] * emb_dim
        assert result.shape == (sum(input_batch_size_per_rank), emb_dim,)
        return result

    @staticmethod
    # pyre-fixme[14]: `backward` overrides method defined in `Function` inconsistently.
    # pyre-fixme[2]: Parameter must be annotated.
    def backward(ctx, grad_output: Tensor) -> Tuple[None, None, Tensor]:

        myreq = ctx.myreq
        a2ai = ctx.a2ai
        pg = ctx.pg
        my_rank = dist.get_rank(pg)

        input_batch_size_per_rank = a2ai.input_batch_size_per_rank
        output_batch_size_per_rank = a2ai.output_batch_size_per_rank

        (input_batch_size_sum, emb_dim) = grad_output.shape
        assert emb_dim == a2ai.emb_dim
        assert input_batch_size_sum == sum(input_batch_size_per_rank)

        sharded_grad_output = grad_output.view(-1)

        sharded_grad_input = torch.empty(
            sum(output_batch_size_per_rank) * emb_dim,
            device=grad_output.device,
            dtype=grad_output.dtype,
        )
        input_split_sizes = [bs * emb_dim for bs in input_batch_size_per_rank]
        output_split_sizes = [bs * emb_dim for bs in output_batch_size_per_rank]

        with record_function("## alltoall_bwd_single ##"):
            req = dist.all_to_all_single(
                output=sharded_grad_input,
                input=sharded_grad_output,
                output_split_sizes=output_split_sizes,
                input_split_sizes=input_split_sizes,
                group=pg,
                async_op=True,
            )

        myreq.req = req
        myreq.tensor = sharded_grad_input
        # Note - this mismatch is by design! We return sharded_grad_output to allow PyTorch shape matching to proceed correctly.
        return (None, None, sharded_grad_output)


class All2All_Pooled_Req(Function):
    @staticmethod
    # pyre-fixme[14]: `forward` overrides method defined in `Function` inconsistently.
    def forward(
        # pyre-fixme[2]: Parameter must be annotated.
        ctx,
        pg: dist.ProcessGroup,
        myreq: Request[Tensor],
        a2ai: All2AllPooledInfo,
        input_embeddings: Tensor,
    ) -> Tensor:
        my_rank = dist.get_rank(pg)
        (B_global, D_local_sum) = input_embeddings.shape

        dim_sum_per_rank = a2ai.dim_sum_per_rank
        batch_size_per_rank = a2ai.batch_size_per_rank
        B_local = batch_size_per_rank[my_rank]
        assert B_global == sum(
            batch_size_per_rank
        ), f"B_global:{B_global} but batch_size_per_rank:{batch_size_per_rank}, sum:{sum(batch_size_per_rank)}"

        sharded_input_embeddings = input_embeddings.view(-1)
        D_global_sum = sum(dim_sum_per_rank)
        sharded_output_embeddings = torch.empty(
            B_local * D_global_sum,
            dtype=input_embeddings.dtype,
            device=input_embeddings.device,
        )
        input_split_sizes = [D_local_sum * B_rank for B_rank in batch_size_per_rank]
        output_split_sizes = [B_local * D_rank_sum for D_rank_sum in dim_sum_per_rank]

        with record_function("## alltoall_fwd_single ##"):
            req = dist.all_to_all_single(
                output=sharded_output_embeddings,
                input=sharded_input_embeddings,
                output_split_sizes=output_split_sizes,
                input_split_sizes=input_split_sizes,
                group=pg,
                async_op=True,
            )
        myreq.req = req
        myreq.tensor = sharded_output_embeddings
        myreq.a2ai = a2ai
        myreq.wait_function = All2All_Pooled_Wait
        ctx.myreq = myreq
        ctx.pg = pg
        return sharded_output_embeddings

    @staticmethod
    # pyre-fixme[2]: Parameter must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def backward(ctx, *unused) -> Tuple[None, None, None, Tensor]:

        pg = ctx.pg
        my_rank = dist.get_rank(pg)
        myreq = ctx.myreq
        a2ai = myreq.a2ai
        myreq.req.wait()
        myreq.req = None
        grad_output = myreq.tensor
        dim_sum_per_rank = a2ai.dim_sum_per_rank
        batch_size_per_rank = a2ai.batch_size_per_rank
        D_local_sum = dim_sum_per_rank[my_rank]
        B_global = sum(batch_size_per_rank)

        grad_input = grad_output.view(B_global, D_local_sum)
        if GRADIENT_DIVISION:
            grad_input.div_(dist.get_world_size(ctx.pg))
        myreq.tensor = None

        return (None, None, None, grad_input)


class All2All_Pooled_Wait(Function):
    @staticmethod
    # pyre-fixme[14]: `forward` overrides method defined in `Function` inconsistently.
    def forward(
        # pyre-fixme[2]: Parameter must be annotated.
        ctx,
        pg: dist.ProcessGroup,
        myreq: Request[Tensor],
        sharded_output_embeddings: Tensor,
    ) -> Tensor:
        my_rank = dist.get_rank(pg)
        a2ai = myreq.a2ai
        ctx.a2ai = a2ai
        myreq.req.wait()
        myreq.req = None
        myreq.tensor = None
        ctx.pg = pg
        ctx.myreq = myreq
        dim_sum_per_rank = a2ai.dim_sum_per_rank
        batch_size_per_rank = a2ai.batch_size_per_rank
        B_local = batch_size_per_rank[my_rank]

        outputs_by_rank = sharded_output_embeddings.split(
            [B_local * D_rank_sum for D_rank_sum in dim_sum_per_rank]
        )
        outputs_by_rank_shape = [output.shape for output in outputs_by_rank]

        result = torch.cat(
            [output.view(B_local, -1) for output in outputs_by_rank], dim=1
        )

        return result

    @staticmethod
    # pyre-fixme[14]: `backward` overrides method defined in `Function` inconsistently.
    # pyre-fixme[2]: Parameter must be annotated.
    def backward(ctx, grad_output: Tensor) -> Tuple[None, None, Tensor]:

        myreq = ctx.myreq
        a2ai = ctx.a2ai
        pg = ctx.pg
        my_rank = dist.get_rank(pg)
        dim_sum_per_rank = a2ai.dim_sum_per_rank
        batch_size_per_rank = a2ai.batch_size_per_rank

        D_local_sum = dim_sum_per_rank[my_rank]
        (B_local, D_global_sum) = grad_output.shape
        B_global = sum(batch_size_per_rank)
        sharded_grad_input_sizes = B_global * D_local_sum

        assert sum(dim_sum_per_rank) == D_global_sum

        sharded_grad_output = _recat_pooled_embedding_grad_out(
            grad_output.contiguous(), dim_sum_per_rank,
        )

        sharded_grad_input = torch.empty(
            sharded_grad_input_sizes, device=grad_output.device, dtype=grad_output.dtype
        )
        output_split_sizes = [
            D_local_sum * B_rank for B_rank in a2ai.batch_size_per_rank
        ]

        with record_function("## alltoall_bwd_single ##"):
            req = dist.all_to_all_single(
                output=sharded_grad_input,
                input=sharded_grad_output,
                output_split_sizes=output_split_sizes,
                input_split_sizes=[
                    B_local * D_rank_sum for D_rank_sum in dim_sum_per_rank
                ],
                group=pg,
                async_op=True,
            )

        myreq.req = req
        myreq.tensor = sharded_grad_input
        # Note - this mismatch is by design! We return sharded_grad_output to allow PyTorch shape matching to proceed correctly.
        return (None, None, sharded_grad_output)


class All2All_Seq_Req(Function):
    @staticmethod
    # pyre-fixme[14]: `forward` overrides method defined in `Function` inconsistently.
    def forward(
        # pyre-fixme[2]: Parameter must be annotated.
        ctx,
        pg: dist.ProcessGroup,
        myreq: Request[Tensor],
        a2ai: All2AllSequenceInfo,
        sharded_input_embeddings: Tensor,
    ) -> Tensor:
        world_size = dist.get_world_size(pg)
        my_rank = dist.get_rank(pg)
        D = a2ai.embedding_dim
        forward_recat_tensor = a2ai.forward_recat_tensor
        lengths_after_sparse_data_all2all = a2ai.lengths_after_sparse_data_all2all * D
        input_splits = [i * D for i in a2ai.output_splits]
        output_splits = [i * D for i in a2ai.input_splits]
        local_T = lengths_after_sparse_data_all2all.shape[0]
        if local_T > 0:
            with record_function("## alltoall_seq_embedding_fwd_permute ##"):
                (
                    permuted_lengths_after_sparse_data_all2all,
                    sharded_input_embeddings,
                    _,
                ) = torch.ops.fbgemm.permute_2D_sparse_data(
                    forward_recat_tensor,
                    lengths_after_sparse_data_all2all.view(local_T * world_size, -1),
                    sharded_input_embeddings.view(-1),
                    None,
                    sharded_input_embeddings.numel(),
                )

        else:
            permuted_lengths_after_sparse_data_all2all = None
        sharded_output_embeddings = torch.empty(
            sum(output_splits),
            dtype=sharded_input_embeddings.dtype,
            device=sharded_input_embeddings.device,
        )

        with record_function("## alltoall_seq_embedding_fwd_single ##"):
            req = dist.all_to_all_single(
                output=sharded_output_embeddings,
                input=sharded_input_embeddings,
                output_split_sizes=output_splits,
                input_split_sizes=input_splits,
                group=pg,
                async_op=True,
            )
        a2ai.permuted_lengths_after_sparse_data_all2all = (
            permuted_lengths_after_sparse_data_all2all
        )
        a2ai.input_splits = input_splits
        a2ai.output_splits = output_splits
        myreq.req = req
        myreq.tensor = sharded_output_embeddings
        myreq.a2ai = a2ai
        myreq.wait_function = All2All_Seq_Req_Wait
        ctx.myreq = myreq
        ctx.pg = pg
        ctx.my_rank = my_rank
        ctx.world_size = world_size
        return sharded_output_embeddings

    @staticmethod
    # pyre-fixme[2]: Parameter must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def backward(ctx, *unused) -> Tuple[None, None, None, Tensor]:
        myreq = ctx.myreq
        a2ai = myreq.a2ai
        D = a2ai.embedding_dim
        backward_recat_tensor = a2ai.backward_recat_tensor
        permuted_lengths_after_sparse_data_all2all = (
            a2ai.permuted_lengths_after_sparse_data_all2all
        )
        myreq.req.wait()
        sharded_grad_input = myreq.tensor
        myreq.req = None
        myreq.tensor = None

        if permuted_lengths_after_sparse_data_all2all is not None:
            with record_function("## alltoall_seq_embedding_bwd_permute ##"):
                _, sharded_grad_input, _ = torch.ops.fbgemm.permute_2D_sparse_data(
                    backward_recat_tensor,
                    permuted_lengths_after_sparse_data_all2all,
                    sharded_grad_input,
                    None,
                    sharded_grad_input.numel(),
                )
        return (None, None, None, sharded_grad_input.view(-1, D))


class All2All_Seq_Req_Wait(Function):
    @staticmethod
    # pyre-fixme[14]: `forward` overrides method defined in `Function` inconsistently.
    def forward(
        # pyre-fixme[2]: Parameter must be annotated.
        ctx,
        pg: dist.ProcessGroup,
        myreq: Request[Tensor],
        sharded_output_embeddings: Tensor,
    ) -> Tensor:
        a2ai = myreq.a2ai
        D = a2ai.embedding_dim
        ctx.a2ai = a2ai
        myreq.req.wait()
        myreq.req = None
        myreq.tensor = None
        ctx.pg = pg
        ctx.myreq = myreq
        return sharded_output_embeddings.view(-1, D)

    @staticmethod
    # pyre-fixme[14]: `backward` overrides method defined in `Function` inconsistently.
    # pyre-fixme[2]: Parameter must be annotated.
    def backward(ctx, sharded_grad_output: Tensor) -> Tuple[None, None, Tensor]:
        myreq = ctx.myreq
        a2ai = ctx.a2ai
        pg = ctx.pg
        input_splits = a2ai.output_splits
        output_splits = a2ai.input_splits
        sharded_grad_input = torch.empty(
            sum(output_splits),
            device=sharded_grad_output.device,
            dtype=sharded_grad_output.dtype,
        )
        with record_function("## alltoall_seq_embedding_bwd_single ##"):
            req = dist.all_to_all_single(
                output=sharded_grad_input,
                input=sharded_grad_output.view(-1),
                output_split_sizes=output_splits,
                input_split_sizes=input_splits,
                group=pg,
                async_op=True,
            )
        myreq.req = req
        myreq.tensor = sharded_grad_input

        # Note - this mismatch is by design! We return sharded_grad_output
        # to allow PyTorch shape matching to proceed correctly.
        return (None, None, sharded_grad_output.view(-1))


class All2Allv_Req(Function):
    @staticmethod
    # pyre-fixme[14]: `forward` overrides method defined in `Function` inconsistently.
    def forward(
        # pyre-fixme[2]: Parameter must be annotated.
        ctx,
        pg: dist.ProcessGroup,
        myreq: Request[Tensor],
        a2ai: All2AllVInfo,
        inputs: List[Tensor],
    ) -> Tensor:
        input_split_sizes = [m * sum(a2ai.D_local_list) for m in a2ai.B_local_list]
        output_split_sizes = [a2ai.B_local * e for e in a2ai.dims_sum_per_rank]
        input = torch.cat(inputs, dim=1).view([-1])
        output = input.new_empty(sum(output_split_sizes))
        with record_function("## alltoallv_bwd_single ##"):
            req = dist.all_to_all_single(
                output,
                input,
                output_split_sizes,
                input_split_sizes,
                group=pg,
                async_op=True,
            )

        myreq.req = req
        myreq.tensor = output
        myreq.wait_function = All2Allv_Wait
        a2ai.input_split_sizes = input_split_sizes
        a2ai.output_split_sizes = output_split_sizes
        myreq.a2ai = a2ai
        ctx.a2ai = a2ai
        ctx.myreq = myreq
        return output

    @staticmethod
    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def backward(ctx, *grad_output):
        a2ai = ctx.a2ai
        myreq = ctx.myreq
        myreq.req.wait()
        myreq.req = None
        grad_input = myreq.tensor
        grad_inputs = grad_input.view([a2ai.B_global, -1]).split(
            a2ai.D_local_list, dim=1
        )
        grad_inputs = [gin.contiguous() for gin in grad_inputs]
        myreq.tensor = None
        return (None, None, None, *grad_inputs)


class All2Allv_Wait(Function):
    @staticmethod
    # pyre-fixme[14]: `forward` overrides method defined in `Function` inconsistently.
    # pyre-fixme[2]: Parameter must be annotated.
    def forward(ctx, pg: dist.ProcessGroup, myreq, output) -> Tuple[Tensor]:
        a2ai = myreq.a2ai
        ctx.a2ai = a2ai
        myreq.req.wait()
        myreq.req = None
        myreq.tensor = None
        ctx.pg = pg
        ctx.myreq = myreq
        outputs = tuple(
            [
                out.view([a2ai.B_local, -1])
                for out in output.split(a2ai.output_split_sizes)
            ]
        )
        return outputs

    @staticmethod
    # pyre-fixme[2]: Parameter must be annotated.
    def backward(ctx, *grad_outputs) -> Tuple[None, None, Tensor]:
        pg = ctx.pg
        myreq = ctx.myreq
        a2ai = ctx.a2ai
        grad_outputs = [gout.contiguous().view([-1]) for gout in grad_outputs]
        grad_output = torch.cat(grad_outputs)
        grad_input = grad_output.new_empty([a2ai.B_global * sum(a2ai.D_local_list)])
        with record_function("## alltoall_bwd_single ##"):
            req = dist.all_to_all_single(
                grad_input,
                grad_output,
                a2ai.input_split_sizes,
                a2ai.output_split_sizes,
                group=pg,
                async_op=True,
            )
        myreq.req = req
        myreq.tensor = grad_input
        return (None, None, grad_output)


class ReduceScatter_Req(Function):
    @staticmethod
    # pyre-fixme[14]: `forward` overrides method defined in `Function` inconsistently.
    def forward(
        # pyre-fixme[2]: Parameter must be annotated.
        ctx,
        pg: dist.ProcessGroup,
        myreq: Request[Tensor],
        rsi: ReduceScatterInfo,
        *inputs: Any,
    ) -> Tensor:
        my_rank = dist.get_rank(pg)
        output = inputs[my_rank].new_empty(
            inputs[my_rank].size(),
            dtype=inputs[my_rank].dtype,
            device=inputs[my_rank].device,
        )

        with record_function("## reduce_scatter ##"):
            req = dist.reduce_scatter(output, list(inputs), group=pg, async_op=True)
        myreq.req = req
        myreq.tensor = output
        myreq.wait_function = ReduceScatter_Wait
        myreq.rsi = rsi
        ctx.myreq = myreq
        ctx.pg = pg
        return output

    @staticmethod
    # pyre-fixme[2]: Parameter must be annotated.
    def backward(ctx, *unused: Tensor) -> Tuple[Optional[Tensor], ...]:
        myreq = ctx.myreq
        myreq.req.wait()
        myreq.req = None
        grad_inputs = list(myreq.tensor)
        # Make it equivalent to running on a single rank.
        if GRADIENT_DIVISION:
            for grad_input in grad_inputs:
                grad_input.div_(dist.get_world_size(ctx.pg))
        myreq.tensor = None
        return (None, None, None, *grad_inputs)


class ReduceScatter_Wait(Function):
    @staticmethod
    # pyre-fixme[14]: `forward` overrides method defined in `Function` inconsistently.
    def forward(
        # pyre-fixme[2]: Parameter must be annotated.
        ctx,
        pg: dist.ProcessGroup,
        myreq: Request[Tensor],
        output: Tensor,
    ) -> Tensor:
        myreq.req.wait()
        myreq.req = None
        myreq.tensor = None
        ctx.myreq = myreq
        ctx.pg = pg
        return output

    @staticmethod
    # pyre-fixme[14]: `backward` overrides method defined in `Function` inconsistently.
    # pyre-fixme[2]: Parameter must be annotated.
    def backward(ctx, grad_output: Tensor) -> Tuple[None, None, Tensor]:
        myreq = ctx.myreq
        rsi = myreq.rsi
        grad_inputs = [
            grad_output.new_empty(
                in_size, dtype=grad_output.dtype, device=grad_output.device,
            )
            for in_size in rsi.input_sizes
        ]
        with record_function("## reduce_scatter_bw (all_gather) ##"):
            req = dist.all_gather(
                grad_inputs, grad_output.contiguous(), group=ctx.pg, async_op=True,
            )
        myreq.req = req
        myreq.tensor = grad_inputs
        return (None, None, grad_output)
