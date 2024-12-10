from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand, tensor


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    # TODO: Implement for Task 4.3.
    # raise NotImplementedError("Need to implement for Task 4.3")

    # we need to first ensure contiguous, then use view

    new_height = height // kh
    new_width = width // kw

    new_tensor = input.contiguous().view(batch, channel, new_height, kh, new_width, kw)

    new_tensor = new_tensor.permute(0, 1, 2, 4, 3, 5)  # makes non-contiguous
    new_tensor = new_tensor.contiguous().view(
        batch, channel, new_height, new_width, kw * kh
    )
    return new_tensor, new_height, new_width


# TODO: Implement for Task 4.3.
def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Tiled average pooling 2D

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width

    """

    batch, channel, height, width = input.shape
    kh, kw = kernel

    # we need to first tile the input tensor
    new_tensor, new_height, new_width = tile(input, kernel)

    # pool using the mean
    pooled = new_tensor.mean(dim=4)

    # use view to make into shape we need
    return pooled.view(batch, channel, new_height, new_width)


# Implement for 4.4


class Max(Function):

    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:
        """max forward function"""
        ctx.save_for_backward(input, dim)
        return input.f.max_reduce(input, int(dim.item()))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """max backward function"""
        input, dim = ctx.saved_values
        grad_output = argmax(input, dim) * grad_output
        return grad_output, 0.0


def max(input: Tensor, dim: int) -> Tensor:
    """Apply max reduction"""
    return Max.apply(input, input._ensure_tensor(dim))


def argmax(input: Tensor, dim: int) -> Tensor:
    """Compute the argmax as a 1-hot tensor"""
    if dim is None:
        out = input.f.max_reduce(input, 0)
    else:
        out = input.f.max_reduce(input, int(input._ensure_tensor(dim).item()))
    return out == input


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Tiled max pooling 2D"""

    batch, channel, height, width = input.shape
    kh, kw = kernel

    # we need to first tile the input tensor
    new_tensor, new_height, new_width = tile(input, kernel)

    # pool using the max
    pooled = max(new_tensor, 4)

    # use view to make into shape we need
    return pooled.contiguous().view(batch, channel, new_height, new_width)


def softmax(input: Tensor, dim: int) -> Tensor:
    """Softmax"""

    exp = input.exp()
    return exp / exp.sum(dim=dim)


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """LogSoftmax"""

    log_sum_exp = (input.exp()).sum(dim).log()
    return input - log_sum_exp


def dropout(input: Tensor, p: float, ignore: bool = False) -> Tensor:
    """Dropout"""

    if ignore:
        return input

    mask = rand(input.shape) > p
    return input * mask
