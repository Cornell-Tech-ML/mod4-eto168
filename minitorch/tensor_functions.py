"""Implementation of the autodifferentiation Functions for Tensor."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np

import minitorch

from . import operators
from .autodiff import Context
from .tensor_ops import SimpleBackend, TensorBackend

if TYPE_CHECKING:
    from typing import Any, List, Tuple

    from .tensor import Tensor
    from .tensor_data import UserIndex, UserShape


def wrap_tuple(x: Any) -> tuple:  # type: ignore
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


# Constructors
class Function:
    @classmethod
    def _backward(cls, ctx: Context, grad_out: Tensor) -> Tuple[Tensor, ...]:
        return wrap_tuple(cls.backward(ctx, grad_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: Tensor) -> Tensor:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: Tensor) -> Tensor:
        """Call the forward function and track history"""
        raw_vals = []
        need_grad = False
        for v in vals:
            if v.requires_grad():
                need_grad = True
            raw_vals.append(v.detach())

        # Create the context.
        ctx = Context(not need_grad)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        # assert isinstance(c, Tensor), "Expected return type Tensor got %s" % (
        #     type(c)
        # )

        # Create a new variable from the result with a new history.
        back = None
        if need_grad:
            back = minitorch.History(cls, ctx, vals)
        return minitorch.Tensor(c._tensor, back, backend=c.backend)


class Neg(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        return t1.f.neg_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        return grad_output.f.neg_map(grad_output)


class Inv(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        ctx.save_for_backward(t1)
        return t1.f.inv_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        (t1,) = ctx.saved_values
        return grad_output.f.inv_back_zip(t1, grad_output)


class Add(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        return t1.f.add_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        return grad_output, grad_output


class All(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Tensor) -> Tensor:
        """Return 1 if all are true"""
        if dim is not None:
            return a.f.mul_reduce(a, int(dim.item()))
        else:
            return a.f.mul_reduce(a.contiguous().view(int(operators.prod(a.shape))), 0)


# TODO: Implement for Task 2.3.
class Mul(Function):
    """
    Tensor Gradients.

    A gradient is a tensor of derivatives.

    #### Simple Case ####

    Let's start with the simple case where a function maps from a tensor to a
    scalar. In this case, the gradient is a tensor of the same shape as the
    input:

    Suppose we have our original function G([x1,x2,x3]) that operates on a
    (3, 1) tensor. If G is Multiplication, then it is a function that maps from
    a tensor-to-scalar.

    In reverse, the gradient, G' is a tensor-to-tensor function that maps from
    a tensor to a tensor. Why is this? Because we need to compute the partial
    derivatives of G with respect to each of the input variables (each one
    defined within the tensor).

    Example: G'([x1,x2,x3]) = x1 * x2 * x3

    G'([x1,x2,x3]) =

    3 total partial derivatves. The notation: G'_x1 means the derivative of the
    G, with respect to x1.

    G'_x1 = x2 * x3
    G'_x2 = x1 * x3
    G'_x3 = x1 * x2

    and the gradiant is: [dG/dx1, dG/dx2, dG/dx3]

    #### Harder Case ####

    Now, let's consider the case where the function maps from a tensor to a
    tensor. In this case, the gradient is a tensor of tensors.

    This is equivalent to a function returning MULTIPLE VALUES.

    Now, suppose:

    G(x) = [G1(x), G2(x), G3(x)],

    where G1, G2, G3 are scalar functionsm and 1, 2, 3 are superscripts
    indicating their index in the tensor.

    For example:

    G([x1, x2]) = [x1, x1 * x2]

    This equation above means G() takes in a tensor of size 2, and returns a
    tensor of size 2. However, G1([x1, x2]) = x1 and G2([x1, x2]) = x1 * x2.
    That is, G1 and G2 are scalar functions that map from a tensor to a scalar.

    Therefore, when we take the derivative of G, we need to take the derivative
    of each of the scalar functions G1, G2, G3 with respect to each of the
    input variables.

    Let's see how this works in practice.

    G'1_x1([x1, x2]) = 1
    G'1_x2([x1, x2]) = 0
    G'2_x1([x1, x2]) = x2
    G'2_x2([x1, x2]) = x1

    The number of derivatives = number of input variables * dimension of the
    output tensor.

    Great. Now let's look at how the chain rule applies.

    suppose we have f(x) is a function that maps a tensor to a scalar.

    ############# MORE DETAIL LATER ################

    Multiplication is a zip, thus the backward function is a zip
    that multiplies d by the gradient

    """

    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        ctx.save_for_backward(t1, t2)
        return t1.f.mul_zip(t1, t2)

    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        t1, t2 = ctx.saved_values
        # critical: remember that the multiplication chain rule REVERSES
        # the operation. Thus, t2 first, t1 second.
        return (
            grad_output.f.mul_zip(t2, grad_output),
            grad_output.f.mul_zip(t1, grad_output),
        )


class Sigmoid(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        out = t1.f.sigmoid_map(t1)
        ctx.save_for_backward(out)  # save sigmoid calculated T1
        return out

    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        # (sigmoid_t1,) = ctx.saved_values
        # # a sigmoid is a map function, so the reverse is a zip.
        # # this is analagous to scalar_functions, where we did:
        # # return sigma * (1.0 - sigma) * d_output
        # return grad_output.f.mul_zip(
        #     sigmoid_t1.f.mul_zip(sigmoid_t1, 1 + sigmoid_t1.f.neg_map(sigmoid_t1)),
        #     grad_output,
        # )
        sigma: Tensor = ctx.saved_values[0]
        return sigma * (-sigma + 1.0) * grad_output


class ReLU(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        ctx.save_for_backward(t1)
        return t1.f.relu_map(t1)

    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        # this is a map, so the reverse is a zip
        (t1,) = ctx.saved_values
        return grad_output.f.relu_back_zip(t1, grad_output)


class Log(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        ctx.save_for_backward(t1)
        return t1.f.log_map(t1)

    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        # this a map, so the reverse is a zip
        (t1,) = ctx.saved_values
        return grad_output.f.log_back_zip(t1, grad_output)


class Exp(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        out = t1.f.exp_map(t1)
        ctx.save_for_backward(out)
        return out

    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        # forward is a map, so backward is a zip
        # e^x * d_output
        (a,) = ctx.saved_values
        return grad_output.f.mul_zip(a, grad_output)


class Sum(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, dim: Tensor) -> Tensor:
        # # follow the scheme of All
        # if dim is not None:
        #     return t1.f.add_reduce(t1, int(dim.item()))
        # else:
        #     return t1.f.add_reduce(
        #         t1.contiguous().view(int(operators.prod(t1.shape))), 0
        #     )
        ctx.save_for_backward(t1.shape, dim)
        return t1.f.add_reduce(t1, int(dim.item()))

    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """
        Reduce is a tricker case. When we reduce, we collapse a dimension.

        Thus, when we backprop, we need to expand the dimension back to the
        original shape.

        Mathematically, for reduce:

        G(x) = sum(x)
        """
        t1_shape, dim = ctx.saved_values
        # we return 0.0 because
        return grad_output, 0.0


class LT(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        ctx.save_for_backward(t1.shape, t2.shape)
        return t1.f.lt_zip(t1, t2)

    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        # the derivative of x < y is 0
        # we need to return 0 tensors of the same shape
        (t1_shape, t2_shape) = ctx.saved_values
        return zeros(t1_shape), zeros(t2_shape)


class EQ(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        ctx.save_for_backward(t1.shape, t2.shape)
        return t1.f.eq_zip(t1, t2)

    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        # the derivative of x == y is 0
        # we need to return 0 tensors of the same shape
        (t1_shape, t2_shape) = ctx.saved_values
        return zeros(t1_shape), zeros(t2_shape)


class IsClose(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        return t1.f.is_close_zip(t1, t2)

    # NO backward


class Permute(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, dims: Tensor) -> Tensor:
        """Permute the dimensions of a tensor."""
        # note, we use t1._tensor because _tensor is TensorData,
        # which is where we implemented permute

        # # dims: Tensor is a tensor of integers that specify the new order
        # # we cannot pass this into permute as it takes integers
        # # thus, we need to convert _storage into a list of integers
        # dims_as_integer = []
        # for i in dims._tensor._storage:
        #     dims_as_integer.append(int(i))

        # # int_order = [int(x) for x in dims._tensor._storage]
        # ctx.save_for_backward(dims_as_integer)

        # # a._new creates a new tensor with the same backend as `a`
        # return t1._new(t1._tensor.permute(*dims_as_integer))
        ctx.save_for_backward(dims)
        return t1._new(t1._tensor.permute(*[int(dims[i]) for i in range(dims.size)]))

    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor]:
        """Permute the gradients back to the original order."""
        # (dims_as_integer,) = ctx.saved_values

        # # since dims_as_integer was the order that we permuted the tensor INTO
        # # we need to permute the gradient back to the original order
        # reverse_dim_dict = {v: i for i, v in enumerate(dims_as_integer)}
        # reverse_dims = [reverse_dim_dict[i] for i in range(len(dims_as_integer))]

        # # return 0 for dims as given on Ed
        # return grad_output._new(grad_output._tensor.permute(*reverse_dims)), 0.0

        order: Tensor = ctx.saved_values[0]
        order2: List[int] = [
            a[0]
            for a in sorted(
                enumerate([order[i] for i in range(order.size)]), key=lambda x: x[1]
            )
        ]
        return grad_output._new(grad_output._tensor.permute(*order2)), 0.0


# Done TODO:2.3


class View(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, shape: Tensor) -> Tensor:
        ctx.save_for_backward(a.shape)
        assert a._tensor.is_contiguous(), "Must be contiguous to view"
        shape2 = [int(shape[i]) for i in range(shape.size)]
        return minitorch.Tensor.make(
            a._tensor._storage, tuple(shape2), backend=a.backend
        )

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Matrix Multiply backward (module 3)"""
        (original,) = ctx.saved_values
        return (
            minitorch.Tensor.make(
                grad_output._tensor._storage, original, backend=grad_output.backend
            ),
            0.0,
        )


class Copy(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        """Id function makes contiguous"""
        return a.f.id_map(a)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Undo"""
        return grad_output


class MatMul(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Matrix Multiply Forward (module 3)"""
        ctx.save_for_backward(t1, t2)
        return t1.f.matrix_multiply(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Matrix Multiply backward (module 3)"""
        t1, t2 = ctx.saved_values

        def transpose(a: Tensor) -> Tensor:
            order = list(range(a.dims))
            order[-2], order[-1] = order[-1], order[-2]
            return a._new(a._tensor.permute(*order))

        return (
            grad_output.f.matrix_multiply(grad_output, transpose(t2)),
            grad_output.f.matrix_multiply(transpose(t1), grad_output),
        )


# Helpers for Constructing tensors
def zeros(shape: UserShape, backend: TensorBackend = SimpleBackend) -> Tensor:
    """Produce a zero tensor of size `shape`.

    Args:
    ----
        shape : shape of tensor
        backend : tensor backend

    Returns:
    -------
        new tensor

    """
    return minitorch.Tensor.make(
        [0.0] * int(operators.prod(shape)), shape, backend=backend
    )


def rand(
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Produce a random tensor of size `shape`.

    Args:
    ----
        shape : shape of tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
    -------
        :class:`Tensor` : new tensor

    """
    vals = [random.random() for _ in range(int(operators.prod(shape)))]
    tensor = minitorch.Tensor.make(vals, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def _tensor(
    ls: Any,
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Produce a tensor with data ls and shape `shape`.

    Args:
    ----
        ls: data for tensor
        shape: shape of tensor
        backend: tensor backend
        requires_grad: turn on autodifferentiation

    Returns:
    -------
        new tensor

    """
    tensor = minitorch.Tensor.make(ls, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def tensor(
    ls: Any, backend: TensorBackend = SimpleBackend, requires_grad: bool = False
) -> Tensor:
    """Produce a tensor with data and shape from ls

    Args:
    ----
        ls: data for tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
    -------
        :class:`Tensor` : new tensor

    """

    def shape(ls: Any) -> List[int]:
        if isinstance(ls, (list, tuple)):
            return [len(ls)] + shape(ls[0])
        else:
            return []

    def flatten(ls: Any) -> List[float]:
        if isinstance(ls, (list, tuple)):
            return [y for x in ls for y in flatten(x)]
        else:
            return [ls]

    cur = flatten(ls)
    shape2 = shape(ls)
    return _tensor(cur, tuple(shape2), backend=backend, requires_grad=requires_grad)


# Gradient check for tensors


def grad_central_difference(
    f: Any, *vals: Tensor, arg: int = 0, epsilon: float = 1e-6, ind: UserIndex
) -> float:
    x = vals[arg]
    up = zeros(x.shape)
    up[ind] = epsilon
    vals1 = [x if j != arg else x + up for j, x in enumerate(vals)]
    vals2 = [x if j != arg else x - up for j, x in enumerate(vals)]
    delta: Tensor = f(*vals1).sum() - f(*vals2).sum()

    return delta[0] / (2.0 * epsilon)


def grad_check(f: Any, *vals: Tensor) -> None:
    """Check whether autodiff matches central difference."""
    for x in vals:
        x.requires_grad_(True)
        x.zero_grad_()
    random.seed(10)
    out = f(*vals)
    out.sum().backward()
    err_msg = """

Gradient check error for function %s.

Input %s

Received derivative %f for argument %d and index %s,
but was expecting derivative %f from central difference.

"""

    for i, x in enumerate(vals):
        ind = x._tensor.sample()
        check = grad_central_difference(f, *vals, arg=i, ind=ind)
        assert x.grad is not None
        np.testing.assert_allclose(
            x.grad[ind],
            check,
            1e-2,
            1e-2,
            err_msg=err_msg % (f, vals, x.grad[ind], i, ind, check),
        )
