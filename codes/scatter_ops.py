import torch


torch.use_deterministic_algorithms(True)


def _output_shape(src, index, dim, dim_size):
    shape = list(src.shape)
    if dim_size is None:
        dim_size = int(index.max().item()) + 1 if index.numel() else 0
    shape[dim] = dim_size
    return shape


def scatter_sum(src, index, dim=0, dim_size=None):
    shape = _output_shape(src, index, dim, dim_size)
    return src.new_zeros(shape).scatter_add_(dim, index, src)


def scatter_mean(src, index, dim=0, dim_size=None):
    total = scatter_sum(src, index, dim, dim_size)
    count = scatter_sum(torch.ones_like(src), index, dim, total.size(dim))
    return total / count.clamp_min(1)


def scatter_softmax(src, index, dim=0, dim_size=None):
    shape = _output_shape(src, index, dim, dim_size)
    maxima = src.new_full(shape, -torch.inf)
    maxima.scatter_reduce_(dim, index, src.detach(), reduce="amax")
    exponentials = (src - maxima.gather(dim, index)).exp()
    totals = scatter_sum(exponentials, index, dim, maxima.size(dim))
    return exponentials / totals.gather(dim, index)
