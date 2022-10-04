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


A = Square()
B = Exp()
C = Square()

# y = exp(x^2)^2
x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)
print(y.data)
