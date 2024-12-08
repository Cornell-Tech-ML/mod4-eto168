from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, Any

import numpy as np
from numba import prange
from numba import njit as _njit

from .tensor_data import (
    MAX_DIMS,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

if TYPE_CHECKING:
    from typing import Callable, Optional

    from .tensor import Tensor
    from .tensor_data import Index, Shape, Storage, Strides

# TIP: Use `NUMBA_DISABLE_JIT=1 pytest tests/ -m task3_1` to run these tests without JIT.

# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
Fn = TypeVar("Fn")


def njit(fn: Fn, **kwargs: Any) -> Fn:
    return _njit(inline="always", **kwargs)(fn)  # type: ignore


to_index = njit(to_index)
index_to_position = njit(index_to_position)
broadcast_index = njit(broadcast_index)


class FastOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        # This line JIT compiles your tensor_map
        f = tensor_map(njit(fn))

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            f(*out.tuple(), *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """See `tensor_ops.py`"""
        f = tensor_zip(njit(fn))

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            f(*out.tuple(), *a.tuple(), *b.tuple())
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """See `tensor_ops.py`"""
        f = tensor_reduce(njit(fn))

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = 1

            # Other values when not sum.
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start

            f(*out.tuple(), *a.tuple(), dim)
            return out

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Batched tensor matrix multiply ::

            for n:
              for i:
                for j:
                  for k:
                    out[n, i, j] += a[n, i, k] * b[n, k, j]

        Where n indicates an optional broadcasted batched dimension.

        Should work for tensor shapes of 3 dims ::

            assert a.shape[-1] == b.shape[-2]

        Args:
        ----
            a : tensor data a
            b : tensor data b

        Returns:
        -------
            New tensor data

        """
        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        tensor_matrix_multiply(*out.tuple(), *a.tuple(), *b.tuple())

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implementations


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """NUMBA low_level tensor_map function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out` and `in` are stride-aligned, avoid indexing

    Args:
    ----
        fn: function mappings floats-to-floats to apply.

    Returns:
    -------
        Tensor map function.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        # TODO: Implement for Task 3.1.
        # raise NotImplementedError("Need to implement for Task 3.1")

        # do not index if stride aligned: defined as if strides and shapes
        # are equal: https://edstem.org/us/courses/62856/discussion/5684711
        # NOTE: in tensor_data.py, we can see that shape and strides are
        # aliases for npt.NDArray, which means that they are numpy arrays.
        # to compare equality, we can use np.array_equal
        if np.array_equal(out_strides, in_strides) and np.array_equal(
            out_shape, in_shape
        ):
            for i in prange(len(out)):
                out[i] = fn(in_storage[i])
        else:
            # otherwise, index
            # convert for loop to use prange for parallelization
            for i in prange(len(out)):
                # use numpy buffers: initialize indices by allocating numpy arrays
                # without initilizing entries.
                # https://edstem.org/us/courses/62856/discussion/5698267
                out_index: Index = np.empty(MAX_DIMS, np.int32)
                in_index: Index = np.empty(MAX_DIMS, np.int32)

                # # proceed with regular indexing
                to_index(i, out_shape, out_index)
                broadcast_index(out_index, out_shape, in_shape, in_index)
                o = index_to_position(out_index, out_strides)
                j = index_to_position(in_index, in_strides)
                out[o] = fn(in_storage[j])

    return njit(_map, parallel=True)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """NUMBA higher-order tensor zip function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out`, `a`, `b` are stride-aligned, avoid indexing

    Args:
    ----
        fn: function maps two floats to float to apply.

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        # TODO: Implement for Task 3.1.
        # raise NotImplementedError("Need to implement for Task 3.1")

        # check if stride aligned
        if (
            np.array_equal(out_strides, a_strides)
            and np.array_equal(out_strides, b_strides)
            and np.array_equal(out_shape, a_shape)
            and np.array_equal(out_shape, b_shape)
        ):
            for i in prange(len(out)):
                out[i] = fn(a_storage[i], b_storage[i])
        else:
            # otherwise, index
            # parallelize main loop
            for i in prange(len(out)):
                out_index: Index = np.zeros(MAX_DIMS, np.int32)
                a_index: Index = np.zeros(MAX_DIMS, np.int32)
                b_index: Index = np.zeros(MAX_DIMS, np.int32)
                to_index(i, out_shape, out_index)
                o = index_to_position(out_index, out_strides)
                broadcast_index(out_index, out_shape, a_shape, a_index)
                j = index_to_position(a_index, a_strides)
                broadcast_index(out_index, out_shape, b_shape, b_index)
                k = index_to_position(b_index, b_strides)
                out[o] = fn(a_storage[j], b_storage[k])

    return njit(_zip, parallel=True)  # type: ignore


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """NUMBA higher-order tensor reduce function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * Inner-loop should not call any functions or write non-local variables

    Args:
    ----
        fn: reduction function mapping two floats to float.

    Returns:
    -------
        Tensor reduce function

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
    ) -> None:
        # TODO: Implement for Task 3.1.
        # raise NotImplementedError("Need to implement for Task 3.1")

        # for reduce, the main loop is parallelized, not the inner loop
        reduce_size = a_shape[reduce_dim]
        for i in prange(len(out)):
            # numpy buffer
            out_index: Index = np.zeros(MAX_DIMS, np.int32)
            to_index(i, out_shape, out_index)
            o = index_to_position(out_index, out_strides)
            # we need to modify inner loop to avoid function calls
            for j in range(reduce_size):
                out_index[reduce_dim] = j
                a = index_to_position(out_index, a_strides)
                out[o] = fn(out[o], a_storage[a])

    return njit(_reduce, parallel=True)  # type: ignore


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """NUMBA tensor matrix multiply function.

    Should work for any tensor shapes that broadcast as long as

    ```
    assert a_shape[-1] == b_shape[-2]
    ```

    Optimizations:

    * Outer loop in parallel
    * No index buffers or function calls
    * Inner loop should have no global writes, 1 multiply.


    Args:
    ----
        out (Storage): storage for `out` tensor
        out_shape (Shape): shape for `out` tensor
        out_strides (Strides): strides for `out` tensor
        a_storage (Storage): storage for `a` tensor
        a_shape (Shape): shape for `a` tensor
        a_strides (Strides): strides for `a` tensor
        b_storage (Storage): storage for `b` tensor
        b_shape (Shape): shape for `b` tensor
        b_strides (Strides): strides for `b` tensor

    Returns:
    -------
        None : Fills in `out`

    """
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0

    # TODO: Implement for Task 3.2.
    # raise NotImplementedError("Need to implement for Task 3.2")

    assert a_shape[-1] == b_shape[-2]

    # we are not allowed to use index buffers or functions
    # to get indexes
    # we will need to use strides and shapes to get the correct data
    # from the storage

    # KEY: here, we working matrix multiplication, so we can assume
    # both a and b are of dimension 2.

    # we iterate for each batch. In this case, the batch should be
    # the first stride, because here we assume dim-2 tensors.
    # and a batch is the rows.
    for batch in prange(out_shape[0]):
        # the `batch` in this case tells us: in the `out` tensor's storage,
        # which "segment" of storage we are working with.

        # for each row of a:
        for arow in range(a_shape[-2]):
            # for each column of b
            for bcol in range(b_shape[-1]):
                # now we need to find the correct values to use in
                # a_storage and b_storage to multiply and sum.
                # what information do we have? we have:
                # the batch
                # a's row number
                # b's column number
                # then, we need to first move to the correct storage position
                # that fits the current batch

                a_position_in_storage = a_batch_stride * batch + arow * a_strides[-2]
                b_position_in_storage = b_batch_stride * batch + bcol * b_strides[-1]

                # compute where to store the position in out
                out_storage_position = (
                    batch * out_strides[0]
                    + arow * out_strides[-2]
                    + bcol * out_strides[-1]
                )

                # now that we have the base storage position, for each row col
                # in
                dot_prod_running_sum = 0.0

                # above, in for arow in range(a_shape[-2]), we iterated
                # over the rows of a, now, we iterate over the columns of a
                # so that we can do the dot product.
                for acol in range(a_shape[-1]):
                    dot_prod_running_sum += (
                        a_storage[a_position_in_storage]
                        * b_storage[b_position_in_storage]
                    )
                    # iterate position using strides. Since we are moving
                    # over the columns (in a), and rows in b we need to adjust
                    # the positions in storage, as defined by the strides.
                    a_position_in_storage += a_strides[-1]
                    b_position_in_storage += b_strides[-2]

                out[out_storage_position] = dot_prod_running_sum


tensor_matrix_multiply = njit(_tensor_matrix_multiply, parallel=True)
assert tensor_matrix_multiply is not None
