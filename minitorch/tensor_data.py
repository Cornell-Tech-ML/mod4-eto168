from __future__ import annotations

import random
from typing import Iterable, Optional, Sequence, Tuple, Union

import numba
import numba.cuda
import numpy as np
import numpy.typing as npt
from numpy import array, float64
from typing_extensions import TypeAlias

from .operators import prod

MAX_DIMS = 32


class IndexingError(RuntimeError):
    """Exception raised for indexing errors."""

    pass


Storage: TypeAlias = npt.NDArray[np.float64]
OutIndex: TypeAlias = npt.NDArray[np.int32]
Index: TypeAlias = npt.NDArray[np.int32]
Shape: TypeAlias = npt.NDArray[np.int32]
Strides: TypeAlias = npt.NDArray[np.int32]

UserIndex: TypeAlias = Sequence[int]
UserShape: TypeAlias = Sequence[int]
UserStrides: TypeAlias = Sequence[int]


def index_to_position(index: Index, strides: Strides) -> int:
    """Converts a multidimensional tensor `index` into a single-dimensional position in
    storage based on strides.

    Args:
        index : index tuple of ints
        strides : tensor strides

    Returns:
        Position in storage

    """
    # TODO: Implement for Task 2.1.

    # Importantly, a tensor, regardless of dimension, is stored in a 1D array
    # of length `size`.
    # The strides are the number of elements to skip to move to the next element
    # in each dimension. For example, in a 2D tensor, the stride in the first
    # dimension is the number of columns. In a 3D tensor, the stride in the first
    # dimension is the number of rows times the number of columns.
    # To summarize: Strides is a tuple that provides the mapping from user
    # indexing to the position in the 1-D storage.

    # the position is the dot product of the index and the strides
    position = 0
    for ind, stride in zip(index, strides):
        position += ind * stride
    return position


def to_index(ordinal: int, shape: Shape, out_index: OutIndex) -> None:
    """Convert an `ordinal` to an index in the `shape`.
    Should ensure that enumerating position 0 ... size of a
    tensor produces every index exactly once. It
    may not be the inverse of `index_to_position`.

    Args:
        ordinal: ordinal position to convert.
        shape : tensor shape.
        out_index : return index corresponding to position.

    """
    # TODO: Implement for Task 2.1.
    # raise NotImplementedError("Need to implement for Task 2.1")

    # here, we convert the ordinal (position) in the 1d array
    # to an index in the tensor, given the shape of the tensor
    # notice we do not return anything. Rather, out_index is provided
    # as an argument and is modified in place.

    # suppose we have a 2x3 matrix, the shape would be (2, 3). The size is 6.
    # the ordinal is a number between 0 and 5. Suppose we are given ordinal 3.
    # if we assume contiguous mapping, then the index would be (1, 0).

    # note: range(len(shape) - 1, -1, -1) traverses shape in reverse order.
    # that is, we start from the last dimension and move to the first dimension.
    # for dim in range(len(shape) - 1, -1, -1):
    #     # the index in the current dimension is the remainder of the ordinal
    #     # why? if we divide the ordinal by the size of the current dim,
    #     # then the remainder tells us how many steps to take (becuase strides
    #     # correspond to how many steps we take per step in a dimension).
    #     out_index[dim] = ordinal % shape[dim]
    #     # update ordinal, as we've "used up" part of it in the
    #     # current dimension
    #     ordinal = ordinal // shape[dim]

    cur_ord = ordinal + 0
    for i in range(len(shape) - 1, -1, -1):
        sh = shape[i]
        out_index[i] = int(cur_ord % sh)
        cur_ord = cur_ord // sh


def broadcast_index(
    big_index: Index, big_shape: Shape, shape: Shape, out_index: OutIndex
) -> None:
    """Convert a `big_index` into `big_shape` to a smaller `out_index`
    into `shape` following broadcasting rules. In this case
    it may be larger or with more dimensions than the `shape`
    given. Additional dimensions may need to be mapped to 0 or
    removed.

    Args:
        big_index : multidimensional index of bigger tensor
        big_shape : tensor shape of bigger tensor
        shape : tensor shape of smaller tensor
        out_index : multidimensional index of smaller tensor

    Returns:
        None

    """
    # TODO: Implement for Task 2.2.

    # suppose we have tensor1 + tensor2 = tensor3
    # we know that tensor3 will be the large tensor.
    # this function asks us, given an index in tensor3,
    # how do we find the corresponding index in tensor1 or tensor2?
    # That is, the smaller tensor can be tensor1 or tensor2. However,
    # given we know the shape of the bigger and smaller, we can already find
    # the where the index in the broadcasted tensor corresponds to in the
    # smaller tensor.

    # # Iterate over the number of dimensions in the smaller shape:
    # # we do this because the smaller shape dictates.
    # # note, we again do reverse iteration.
    # for dimension in range(len(shape) - 1, -1, -1):
    #     # print(dimension)
    #     # here, we implement rule 1. If the dimension in the smaller shape
    #     # is 1, then this dimension in the smaller shape is copied n times.
    #     # it is "stretched" to match that of the larger shape.
    #     if shape[dimension] == 1:
    #         out_index[dimension] = 0
    #     else:
    #         # why is this the case? Suggested by copilot.
    #         # t
    #         out_index[dimension] = big_index[dimension + len(big_shape) - len(shape)]

    for i, s in enumerate(shape):
        if s > 1:
            out_index[i] = big_index[i + (len(big_shape) - len(shape))]
        else:
            out_index[i] = 0
    return None


def shape_broadcast(shape1: UserShape, shape2: UserShape) -> UserShape:
    """Broadcast two shapes to create a new union shape.

    Args:
        shape1 : first shape
        shape2 : second shape

    Returns:
        broadcasted shape

    Raises:
        IndexingError : if cannot broadcast

    """
    # TODO: Implement for Task 2.2.
    # raise NotImplementedError("Need to implement for Task 2.2")

    # In this case, give the shape of two tensors, we want to find the shape
    # of the tensor that would result from broadcasting the two tensors.

    # The shape of the broadcasted tensor is the maximum of the two shapes
    # in each dimension. If the two shapes are not equal in a dimension, then
    # the shape of the broadcasted tensor is the maximum of the two shapes.
    # However, because our rule is we can add a dimension of shape 1 to the
    # left, if the

    # new_shape = []
    # shape1_dims = len(shape1)
    # shape2_dims = len(shape2)

    # # print("shape1_dims", shape1_dims)
    # # print("shape2_dims", shape2_dims)

    # # we iterate using the largest tensor
    # if shape1_dims > shape2_dims:
    #     big_tensor = shape1
    #     small_tensor = shape2
    # else:
    #     big_tensor = shape2
    #     small_tensor = shape1

    # num_dims_iterated = 0
    # # again, reverse iterate, starting from the rightmost dimension.
    # # that is, if shape is (1, 2, 3), we start at 3, 2, then 1.
    # for dimension in range(len(big_tensor) - 1, -1, -1):
    #     # print("Dimension", dimension)

    #     # here, we iterate. When we iterate more dimensions than the smaller
    #     # tensor, then we just take the shape of the bigger tensor.
    #     num_dims_iterated += 1
    #     # print("num_dims_iterated", num_dims_iterated)

    #     # shift the index of the smaller tensor
    #     small_tensor_index = dimension - (len(big_tensor) - len(small_tensor))
    #     # print("small_tensor_index", small_tensor_index)

    #     # print("dimension + 1", dimension + 1)
    #     # print("len(small_tensor) - 1", len(small_tensor) - 1)
    #     if num_dims_iterated > len(small_tensor):
    #         # if bigger tensor has more dimensions than the smaller tensor
    #         # then for those extra dimensions, we can "add" a dimension of
    #         # shape 1 for the smaller tensor. This is equivalent to taking the
    #         # shape of the bigger tensor.

    #         # print("big_tensor[dimension]", big_tensor[dimension])
    #         new_shape.append(big_tensor[dimension])
    #         # print("new_shape", new_shape)
    #         continue

    #     # can't broadcast condition
    #     # if the shapes are not equal in a dimension, and neither shape is 1,
    #     # then the shapes cannot be broadcasted.
    #     if (
    #         big_tensor[dimension] != small_tensor[small_tensor_index]
    #         and big_tensor[dimension] != 1
    #         and small_tensor[small_tensor_index] != 1
    #     ):
    #         raise IndexingError(f"Shapes {shape1} and {shape2} are not broadcastable.")

    #     # if either shape has 1, then the broadcasted shape is the maximum
    #     if big_tensor[dimension] == 1 or small_tensor[small_tensor_index] == 1:
    #         new_shape.append(
    #             max(big_tensor[dimension], small_tensor[small_tensor_index])
    #         )
    #     # if the shapes are equal, then the broadcasted shape is the same
    #     elif big_tensor[dimension] == small_tensor[small_tensor_index]:
    #         new_shape.append(big_tensor[dimension])

    #     # print("new_shape", new_shape)

    # return tuple(reversed(new_shape))

    a, b = shape1, shape2
    m = max(len(a), len(b))
    c_rev = [0] * m
    a_rev = list(reversed(a))
    b_rev = list(reversed(b))
    for i in range(m):
        if i >= len(a):
            c_rev[i] = b_rev[i]
        elif i >= len(b):
            c_rev[i] = a_rev[i]
        else:
            c_rev[i] = max(a_rev[i], b_rev[i])
            if a_rev[i] != c_rev[i] and a_rev[i] != 1:
                raise IndexingError(f"Broadcast failure {a} {b}")
            if b_rev[i] != c_rev[i] and b_rev[i] != 1:
                raise IndexingError(f"Broadcast failure {a} {b}")
    return tuple(reversed(c_rev))


def strides_from_shape(shape: UserShape) -> UserStrides:
    """Return a contiguous stride for a shape"""
    layout = [1]
    offset = 1
    for s in reversed(shape):
        layout.append(s * offset)
        offset = s * offset
    return tuple(reversed(layout[:-1]))


class TensorData:
    _storage: Storage
    _strides: Strides
    _shape: Shape
    strides: UserStrides
    shape: UserShape
    dims: int

    def __init__(
        self,
        storage: Union[Sequence[float], Storage],
        shape: UserShape,
        strides: Optional[UserStrides] = None,
    ):
        if isinstance(storage, np.ndarray):
            self._storage = storage
        else:
            self._storage = array(storage, dtype=float64)

        if strides is None:
            strides = strides_from_shape(shape)

        assert isinstance(strides, tuple), "Strides must be tuple"
        assert isinstance(shape, tuple), "Shape must be tuple"
        if len(strides) != len(shape):
            raise IndexingError(f"Len of strides {strides} must match {shape}.")
        self._strides = array(strides)
        self._shape = array(shape)
        self.strides = strides
        self.dims = len(strides)
        self.size = int(prod(shape))
        self.shape = shape
        assert len(self._storage) == self.size

    def to_cuda_(self) -> None:  # pragma: no cover
        """Convert to cuda"""
        if not numba.cuda.is_cuda_array(self._storage):
            self._storage = numba.cuda.to_device(self._storage)

    def is_contiguous(self) -> bool:
        """Check that the layout is contiguous, i.e. outer dimensions have bigger strides than inner dimensions.

        Returns:
            bool : True if contiguous

        """
        last = 1e9
        for stride in self._strides:
            if stride > last:
                return False
            last = stride
        return True

    @staticmethod
    def shape_broadcast(shape_a: UserShape, shape_b: UserShape) -> UserShape:
        return shape_broadcast(shape_a, shape_b)

    def index(self, index: Union[int, UserIndex]) -> int:
        if isinstance(index, int):
            aindex: Index = array([index])
        else:  # if isinstance(index, tuple):
            aindex = array(index)

        # Pretend 0-dim shape is 1-dim shape of singleton
        shape = self.shape
        if len(shape) == 0 and len(aindex) != 0:
            shape = (1,)

        # Check for errors
        if aindex.shape[0] != len(self.shape):
            raise IndexingError(f"Index {aindex} must be size of {self.shape}.")
        for i, ind in enumerate(aindex):
            if ind >= self.shape[i]:
                raise IndexingError(f"Index {aindex} out of range {self.shape}.")
            if ind < 0:
                raise IndexingError(f"Negative indexing for {aindex} not supported.")

        # Call fast indexing.
        return index_to_position(array(index), self._strides)

    def indices(self) -> Iterable[UserIndex]:
        lshape: Shape = array(self.shape)
        out_index: Index = array(self.shape)
        for i in range(self.size):
            to_index(i, lshape, out_index)
            yield tuple(out_index)

    def sample(self) -> UserIndex:
        """Get a random valid index"""
        return tuple((random.randint(0, s - 1) for s in self.shape))

    def get(self, key: UserIndex) -> float:
        x: float = self._storage[self.index(key)]
        return x

    def set(self, key: UserIndex, val: float) -> None:
        self._storage[self.index(key)] = val

    def tuple(self) -> Tuple[Storage, Shape, Strides]:
        """Return core tensor data as a tuple."""
        return (self._storage, self._shape, self._strides)

    def permute(self, *order: int) -> TensorData:
        """Permute the dimensions of the tensor.

        Args:
            *order: a permutation of the dimensions

        Returns:
            New `TensorData` with the same storage and a new dimension order.

        """
        assert list(sorted(order)) == list(
            range(len(self.shape))
        ), f"Must give a position to each dimension. Shape: {self.shape} Order: {order}"

        # # TODO: Implement for Task 2.1.
        # # permutating a tensor is arbitrarily reorderin the dimensions of the
        # # input tensor. That is, given a matrix (2d tensor) of shape (2, 5)
        # # with 2 rows, 5 columns, calling permute(1, 0) would return a new
        # # matrix of shape (5, 2) with 5 rows and 2 columns.
        # # Permute is saying, put the 1st dimension of the input tensor in the
        # # 0th dimension of the output tensor, and the 0th dimension of the
        # # input tensor in the 1st dimension of the output tensor.

        # # it is important to note that this does not change the data in the tensor,
        # # it only changes the way the data is indexed.
        # # for the new shape, order (1, 0) means take the first dimension
        # # to be the 0th, and the 0th, to be the first.

        # new_shape = []
        # new_stride = []
        # for int in order:
        #     new_shape.append(self.shape[int])
        #     # update the order of the strides
        #     new_stride.append(self._strides[int])

        # # debug prints
        # # print("old shape", self.shape)
        # # print("order is", order)
        # # print("new shape", new_shape)
        # # print("------------------------")

        # # make a new tensorData with same storage, but new shape
        # # and strides
        # newTensorData = TensorData(self._storage, tuple(new_shape), tuple(new_stride))

        # return newTensorData

        return TensorData(
            self._storage,
            tuple([self.shape[i] for i in order]),
            tuple([self._strides[i] for i in order]),
        )

    def to_string(self) -> str:
        """Convert to string"""
        s = ""
        for index in self.indices():
            l = ""
            for i in range(len(index) - 1, -1, -1):
                if index[i] == 0:
                    l = "\n%s[" % ("\t" * i) + l
                else:
                    break
            s += l
            v = self.get(index)
            s += f"{v:3.2f}"
            l = ""
            for i in range(len(index) - 1, -1, -1):
                if index[i] == self.shape[i] - 1:
                    l += "]"
                else:
                    break
            if l:
                s += l
            else:
                s += " "
        return s
