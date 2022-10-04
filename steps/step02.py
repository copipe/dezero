from abc import ABCMeta, abstractmethod

import numpy as np


class Variable:
    """A kind of Tensor that is to be considered a module parameter.

    Args:
        data (Tensor): parameter tensor.
    """

    def __init__(self, data: np.ndarray):
        self.data = data


class Function(metaclass=ABCMeta):
    """Base class to create custom Function"""

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


x = Variable(np.array(10))
f = Square()
y = f(x)
print(type(y))
print(y.data)
