"""Collection of the core mathematical operators
used throughout the code base.
"""

import math
from typing import Callable, Iterable

# ## Task 0.1

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if
# x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.


def mul(x: float, y: float) -> float:
    """multiply

    Args:
    ----
        x (float): first float
        y (float): second float

    Returns:
    -------
        float: product of x and y

    """
    return x * y


def id(x: float) -> float:
    """id

    Args:
    ----
        x (float): input

    Returns:
    -------
        float: x

    """
    return x


def add(x: float, y: float) -> float:
    """Add

    Args:
    ----
        x (float): Num1
        y (float): Num2

    Returns:
    -------
        float: sum of x and y

    """
    return x + y


def neg(x: float) -> float:
    """Find negative of x"""
    return -x


def lt(x: float, y: float) -> float:
    """Check if x is less than y"""
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    """Check if x is equal to y"""
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """Find max of x and y"""
    return x if x > y else y


def is_close(x: float, y: float) -> float:
    """Check if x is close to y"""
    return (x - y < 1e-2) and (y - x < 1e-2)


def sigmoid(x: float) -> float:
    """Sigmoid function"""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Rectified Linear Unit"""
    return x if x > 0 else 0.0


EPS = 1e-6


def log(x: float) -> float:
    """Logarithm"""
    return math.log(x + EPS)


def exp(x: float) -> float:
    """Exponential"""
    return math.exp(x)


def log_back(x: float, d: float) -> float:
    """Logarithm backprop"""
    return d / (x + EPS)


def inv(x: float) -> float:
    """Inverse"""
    return 1.0 / x


def inv_back(x: float, d: float) -> float:
    """Inverse backprop"""
    return d * (-1.0 / x**2)


def relu_back(x: float, d: float) -> float:
    """ReLU backprop"""
    return d if x > 0 else 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.
def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """Higher order map"""

    def _map(ls: Iterable[float]) -> Iterable[float]:
        ret = []
        for x in ls:
            ret.append(fn(x))
        return ret

    return _map


def zipWith(
    fn: Callable[[float, float], float]
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """Zip two lists with a function"""

    def _zipWith(xs: Iterable[float], ys: Iterable[float]) -> Iterable[float]:
        ret = []
        for x, y in zip(xs, ys):
            ret.append(fn(x, y))
        return ret

    return _zipWith


def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    """Reduce a list"""

    def _reduce(ls: Iterable[float]) -> float:
        ret = start
        for x in ls:
            ret = fn(ret, x)
        return ret

    return _reduce


def addLists(list1: Iterable[float], list2: Iterable[float]) -> Iterable[float]:
    """Add two lists"""
    return zipWith(add)(list1, list2)


def negList(list: Iterable[float]) -> Iterable[float]:
    """Negate a list"""
    return map(neg)(list)


def sum(list: Iterable[float]) -> float:
    """Sum of a list"""
    return reduce(add, 0.0)(list)


def prod(list: Iterable[float]) -> float:
    """Product of a list"""
    return reduce(mul, 1.0)(list)
