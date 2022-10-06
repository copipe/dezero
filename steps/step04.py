from abc import ABCMeta, abstractmethod
from typing import Callable, Union

import numpy as np


class Variable:
    """A kind of Tensor that is to be considered a module parameter.

    Args:
        data (Tensor): parameter tensor.
    """

    def __init__(self, data: np.ndarray):
        self.data = data


class Function(metaclass=ABCMeta):
    """Base class to create custom Function
    `__call__` applies the operation defined by `forward` to input x (Variable)
    """

    def __call__(self, input: Variable) -> Variable:
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        return output

    @abstractmethod
    def forward(self, x: np.ndarray):
        pass


class Square(Function):
    """Perform a square operation."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x**2


class Exp(Function):
    """Perform a exponential operation."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.exp(x)


def numerical_diff(
    f: Union[Callable[[Variable], Variable], Function], x: Variable, eps: float = 1e-4
) -> float:
    """Numerical differentiation by central difference approximation

    Args:
        f : Function to apply
        x : input tensor
        exp : small differences used for numerical differentiatoin. Defaults to 1e-4.

    Returns:
        _type_: _description_
    """
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)


f1 = Square()
x = Variable(np.array(2.0))
dy = numerical_diff(f1, x)
print(dy)


def f2(x):
    A = Square()
    B = Exp()
    C = Square()
    return C(B(A(x)))


x = Variable(np.array(0.5))
dy = numerical_diff(f2, x)
print(dy)
