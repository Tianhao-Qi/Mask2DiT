import torch
import torch.distributed as dist

# Global variable to cache created SP group
_SP_GROUP = None

def init_sp_group(sequence_parallel_size):
    """
    Initializes sequence parallel group.

    Parameters:
    - sequence_parallel_size (int): Number of ranks per SP group
    """
    global _SP_GROUP

    if _SP_GROUP is not None:
        return _SP_GROUP

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    assert world_size % sequence_parallel_size == 0, \
        f"World size {world_size} must be divisible by sequence_parallel_size {sequence_parallel_size}"

    group_id = rank // sequence_parallel_size
    ranks_in_group = list(range(
        group_id * sequence_parallel_size,
        (group_id + 1) * sequence_parallel_size
    ))

    _SP_GROUP = dist.new_group(ranks=ranks_in_group)

    if dist.get_rank(_SP_GROUP) == 0:
        print(f"[SP] Initialized SP group {group_id} with ranks {ranks_in_group}")

    return _SP_GROUP


def get_sp_group():
    """Returns the current process's sequence parallel group"""
    assert _SP_GROUP is not None, "SP group not initialized, call init_sp_group() first"
    return _SP_GROUP


def get_sequence_parallel_rank():
    """Returns the rank of the current process within the SP group"""
    return dist.get_rank(get_sp_group())


def get_sequence_parallel_world_size():
    """Returns the size (number of ranks) of the SP group"""
    return dist.get_world_size(get_sp_group())


# ======================================================
# AlltoAll
# ======================================================


def _all_to_all_func(input_, world_size, group, scatter_dim, gather_dim):
    input_list = [t.contiguous() for t in torch.tensor_split(input_, world_size, scatter_dim)]
    output_list = [torch.empty_like(input_list[0]) for _ in range(world_size)]
    dist.all_to_all(output_list, input_list, group=group)
    return torch.cat(output_list, dim=gather_dim).contiguous()


class _AllToAll(torch.autograd.Function):
    """All-to-all communication.

    Args:
        input_: input matrix
        process_group: communication group
        scatter_dim: scatter dimension
        gather_dim: gather dimension
    """

    @staticmethod
    def forward(ctx, input_, process_group, scatter_dim, gather_dim):
        ctx.process_group = process_group
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        world_size = dist.get_world_size(process_group)

        return _all_to_all_func(input_, world_size, process_group, scatter_dim, gather_dim)

    @staticmethod
    def backward(ctx, *grad_output):
        process_group = ctx.process_group
        scatter_dim = ctx.gather_dim
        gather_dim = ctx.scatter_dim
        return_grad = _AllToAll.apply(*grad_output, process_group, scatter_dim, gather_dim)
        return (return_grad, None, None, None)


def all_to_all_comm(input_, process_group=None, scatter_dim=2, gather_dim=1):
    return _AllToAll.apply(input_, process_group, scatter_dim, gather_dim)


# ======================================================
# Sequence Gather & Split
# ======================================================


def _split_sequence_func(input_, pg: dist.ProcessGroup, dim: int, pad: int):
    # skip if only one rank involved
    world_size = dist.get_world_size(pg)
    rank = dist.get_rank(pg)
    if world_size == 1:
        return input_

    if pad > 0:
        pad_size = list(input_.shape)
        pad_size[dim] = pad
        input_ = torch.cat([input_, torch.zeros(pad_size, dtype=input_.dtype, device=input_.device)], dim=dim)

    dim_size = input_.size(dim)
    assert dim_size % world_size == 0, f"dim_size ({dim_size}) is not divisible by world_size ({world_size})"

    tensor_list = torch.split(input_, dim_size // world_size, dim=dim)
    output = tensor_list[rank].contiguous()
    return output


def _gather_sequence_func(input_, pg: dist.ProcessGroup, dim: int, pad: int):
    # skip if only one rank involved
    input_ = input_.contiguous()
    world_size = dist.get_world_size(pg)
    dist.get_rank(pg)

    if world_size == 1:
        return input_

    # all gather
    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    #assert input_.device.type == "cuda"
    torch.distributed.all_gather(tensor_list, input_, group=pg)

    # concat
    output = torch.cat(tensor_list, dim=dim)

    if pad > 0:
        output = output.narrow(dim, 0, output.size(dim) - pad)

    return output


class _GatherForwardSplitBackward(torch.autograd.Function):
    """
    Gather the input sequence.

    Args:
        input_: input matrix.
        process_group: process group.
        dim: dimension
    """

    @staticmethod
    def symbolic(graph, input_):
        return _gather_sequence_func(input_)

    @staticmethod
    def forward(ctx, input_, process_group, dim, grad_scale, pad):
        ctx.process_group = process_group
        ctx.dim = dim
        ctx.grad_scale = grad_scale
        ctx.pad = pad
        return _gather_sequence_func(input_, process_group, dim, pad)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.grad_scale == "up":
            grad_output = grad_output * dist.get_world_size(ctx.process_group)
        elif ctx.grad_scale == "down":
            grad_output = grad_output / dist.get_world_size(ctx.process_group)

        return _split_sequence_func(grad_output, ctx.process_group, ctx.dim, ctx.pad), None, None, None, None


class _SplitForwardGatherBackward(torch.autograd.Function):
    """
    Split sequence.

    Args:
        input_: input matrix.
        process_group: parallel mode.
        dim: dimension
    """

    @staticmethod
    def symbolic(graph, input_):
        return _split_sequence_func(input_)

    @staticmethod
    def forward(ctx, input_, process_group, dim, grad_scale, pad):
        ctx.process_group = process_group
        ctx.dim = dim
        ctx.grad_scale = grad_scale
        ctx.pad = pad
        return _split_sequence_func(input_, process_group, dim, pad)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.grad_scale == "up":
            grad_output = grad_output * dist.get_world_size(ctx.process_group)
        elif ctx.grad_scale == "down":
            grad_output = grad_output / dist.get_world_size(ctx.process_group)

        if hasattr(ctx, 'pad') and ctx.pad is not None:
            #print('ctx.pad', ctx.pad)
            return _gather_sequence_func(grad_output, ctx.process_group, ctx.dim, ctx.pad), None, None, None, None


def split_sequence(input_, process_group, dim, grad_scale=1.0, pad=0):
    return _SplitForwardGatherBackward.apply(input_, process_group, dim, grad_scale, pad)


def gather_sequence(input_, process_group, dim, grad_scale=1.0, pad=0):
    return _GatherForwardSplitBackward.apply(input_, process_group, dim, grad_scale, pad)


# ==============================
# Pad
# ==============================

PAD_DICT = {}


def set_pad(name: str, dim_size: int, parallel_group: dist.ProcessGroup):
    sp_size = dist.get_world_size(parallel_group)
    pad = (sp_size - (dim_size % sp_size)) % sp_size
    global PAD_DICT
    PAD_DICT[name] = pad


def get_pad(name) -> int:
    return PAD_DICT[name]