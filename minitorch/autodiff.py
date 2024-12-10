from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple, Protocol


# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    # TODO: Implement for Task 1.1.

    # note, the notation *vals means that vals is a tuple of values,
    # where there is no restriction on the number of positional arguments.

    # following the formula given on the wikipedia page,
    # for multivariate functions (suppose, f(x, y)),
    # the central difference with respect to ONE argument is:
    # f_x(x + h, y) - f_x(x - h, y)
    # therefore, we pass all arguments UP TO [arg], then
    # extract the argument at the arg index,
    # and modify that by epsilon/2.
    # Then we pass the remaining arguments.
    # -------------------------------
    # component_1 = f([vals[:arg], vals[arg] + epsilon, vals[arg + 1 :]])
    # component_2 = f([vals[:arg], vals[arg] - epsilon, vals[arg + 1 :]])

    vals1 = [v for v in vals]
    vals2 = [v for v in vals]
    vals1[arg] = vals1[arg] + epsilon
    vals2[arg] = vals2[arg] - epsilon

    delta = f(*vals1) - f(*vals2)
    return delta / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None: ...

    """Accumulates a derivative from a leaf node."""

    @property
    def unique_id(self) -> int: ...

    def is_leaf(self) -> bool: ...

    """Returns True if the variable is a leaf node."""

    def is_constant(self) -> bool: ...

    """Returns True if the variable is a constant node. """

    @property
    def parents(self) -> Iterable["Variable"]: ...

    """Returns the parents of the variable."""

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]: ...

    """Applies the chain rule to get the derivative of the parents."""


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """
    # TODO: Implement for Task 1.4.

    # Topo sort logic:
    # topological sort is a graph traversal method of a directed graph
    # and produces a linear ordering of its vertices such that for every
    # directed edge (u, v) from vertex u to vertex v, u comes before v in the
    # ordering.

    order: List[Variable] = []
    seen = set()

    def visit(var: Variable):
        if var.unique_id in seen or var.is_constant():
            return

        if not var.is_leaf():
            for m in var.parents:
                if not m.is_constant():
                    visit(m)

        seen.add(var.unique_id)
        order.insert(0, var)

    visit(variable)
    return order


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
    ----
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.

    """
    # TODO: Implement for Task 1.4.
    queue = topological_sort(variable)
    derivatives = {}

    derivatives[variable.unique_id] = deriv
    for var in queue:
        deriv = derivatives[var.unique_id]
        if var.is_leaf():
            var.accumulate_derivative(deriv)
        else:
            for v, d in var.chain_rule(deriv):
                if v.is_constant():
                    continue
                derivatives.setdefault(v.unique_id, 0.0)
                derivatives[v.unique_id] = derivatives[v.unique_id] + d


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
