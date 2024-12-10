import pytest
from hypothesis import given

import minitorch
from minitorch import Tensor

from .strategies import assert_close
from .tensor_strategies import tensors


@pytest.mark.task4_3
@given(tensors(shape=(1, 1, 4, 4)))
def test_avg(t: Tensor) -> None:
    out = minitorch.avgpool2d(t, (2, 2))
    assert_close(
        out[0, 0, 0, 0], sum([t[0, 0, i, j] for i in range(2) for j in range(2)]) / 4.0
    )

    out = minitorch.avgpool2d(t, (2, 1))
    assert_close(
        out[0, 0, 0, 0], sum([t[0, 0, i, j] for i in range(2) for j in range(1)]) / 2.0
    )

    out = minitorch.avgpool2d(t, (1, 2))
    assert_close(
        out[0, 0, 0, 0], sum([t[0, 0, i, j] for i in range(1) for j in range(2)]) / 2.0
    )
    minitorch.grad_check(lambda t: minitorch.avgpool2d(t, (2, 2)), t)


@pytest.mark.task4_4
@given(tensors(shape=(2, 3, 4)))
def test_max(t: Tensor) -> None:
    # TODO: Implement for Task 4.4.
    # raise NotImplementedError("Need to implement for Task 4.4")

    # run max on each dimension
    out_dim0 = minitorch.nn.max(t, 0)
    out_dim1 = minitorch.nn.max(t, 1)
    out_dim2 = minitorch.nn.max(t, 2)

    print(out_dim0.shape)

    # Check it matches expectations:
    # here, we use logic from test_max_pool

    # check for dimension 0, 1, and then 2
    # that is, take the max value in each dimension, then sure our max()
    # function gets the same value
    assert out_dim0[0, 0, 0] == max([t[i, 0, 0] for i in range(t.shape[0])])
    assert out_dim1[0, 0, 0] == max([t[0, i, 0] for i in range(t.shape[1])])
    assert out_dim2[0, 0, 0] == max([t[0, 0, i] for i in range(t.shape[2])])

    # test backward, as taken from test_softmax
    # I chose to do t + minitorch.rand(t.shape) as the input
    # because copilot told me:

    # Ensuring Non-Trivial Gradients:

    # If the input tensor t has trivial values (e.g., all elements are the
    # same), the gradient might be zero or have a simple pattern.
    # Adding noise ensures that the gradients are non-trivial and the
    # gradient check is more meaningful.

    # after the falsifying example was a tensor of all 0s when I just used
    # t. It also failed when I tried t + 1.
    minitorch.grad_check(
        lambda a: minitorch.nn.max(a, dim=2), t + minitorch.rand(t.shape)
    )


@pytest.mark.task4_4
@given(tensors(shape=(1, 1, 4, 4)))
def test_max_pool(t: Tensor) -> None:
    out = minitorch.maxpool2d(t, (2, 2))
    print(out)
    print(t)
    assert_close(
        out[0, 0, 0, 0], max([t[0, 0, i, j] for i in range(2) for j in range(2)])
    )

    out = minitorch.maxpool2d(t, (2, 1))
    assert_close(
        out[0, 0, 0, 0], max([t[0, 0, i, j] for i in range(2) for j in range(1)])
    )

    out = minitorch.maxpool2d(t, (1, 2))
    assert_close(
        out[0, 0, 0, 0], max([t[0, 0, i, j] for i in range(1) for j in range(2)])
    )


@pytest.mark.task4_4
@given(tensors())
def test_drop(t: Tensor) -> None:
    q = minitorch.dropout(t, 0.0)
    idx = q._tensor.sample()
    assert q[idx] == t[idx]
    q = minitorch.dropout(t, 1.0)
    assert q[q._tensor.sample()] == 0.0
    q = minitorch.dropout(t, 1.0, ignore=True)
    idx = q._tensor.sample()
    assert q[idx] == t[idx]


@pytest.mark.task4_4
@given(tensors(shape=(1, 1, 4, 4)))
def test_softmax(t: Tensor) -> None:
    q = minitorch.softmax(t, 3)
    x = q.sum(dim=3)
    assert_close(x[0, 0, 0, 0], 1.0)

    q = minitorch.softmax(t, 1)
    x = q.sum(dim=1)
    assert_close(x[0, 0, 0, 0], 1.0)

    minitorch.grad_check(lambda a: minitorch.softmax(a, dim=2), t)


@pytest.mark.task4_4
@given(tensors(shape=(1, 1, 4, 4)))
def test_log_softmax(t: Tensor) -> None:
    q = minitorch.softmax(t, 3)
    q2 = minitorch.logsoftmax(t, 3).exp()
    for i in q._tensor.indices():
        assert_close(q[i], q2[i])

    minitorch.grad_check(lambda a: minitorch.logsoftmax(a, dim=2), t)
