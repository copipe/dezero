from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Optional

import numpy as np


class Variable:
    """A kind of Tensor that is to be considered a module parameter.

    Attributes:
        data (np.ndarray) : parameter tensor.
        grad (float): gradient computed by backpropagation.
        creator (Function): The function that created this variable.
    """

    def __init__(self, data: np.ndarray):
        self.data = data
        self.grad: Optional[np.ndarray] = None
        self.creator: Optional[Function] = None

    def set_creator(self, func: Function):
        self.creator = func

    def backward(self):
        f = self.creator
        if f is not None:
            x = f.input
            x.grad = f.backward(self.grad)
            x.backward()


class Function(metaclass=ABCMeta):
    """Base class to create custom Function
    `__call__` applies the operation defined by `forward` to input x (Variable)

    Attributes:
        input (Variable): input during forward propagation
    """

    def __call__(self, input: Variable) -> Variable:
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        output.set_creator(self)
        self.input = input
        self.output = output
        return output

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, gy: np.ndarray) -> np.ndarray:
        pass


class Square(Function):
    """Perform a square operation."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x**2

    def backward(self, gy: np.ndarray) -> np.ndarray:
        x = self.input.data
        gx = 2 * x * gy
        return gx


class Exp(Function):
    """Perform a exponential operation."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.exp(x)

    def backward(self, gy: np.ndarray) -> np.ndarray:
        x = self.input.data
        gx = np.exp(x) * gy
        return gx


def numerical_diff(f: Function, x: Variable, eps: float = 1e-4) -> float:
    """Numerical differentiation by central difference approximation

    Args:
        f (Function) : Function to apply
        x (Variable) : Input tensor
        exp (float) : Small differences used for numerical differentiatoin. Defaults to 1e-4.

    Returns:
        float :  Numerical differentiation
    """
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)


A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)

assert y.creator == C
assert y.creator.input == b
assert y.creator.input.creator == B
assert y.creator.input.creator.input == a
assert y.creator.input.creator.input.creator == A
assert y.creator.input.creator.input.creator.input == x

y.grad = np.array(1.0)
C_ = y.creator
b = C_.input
b.grad = C_.backward(y.grad)

B_ = b.creator
assert B_ is not None
a = B_.input
a.grad = B_.backward(b.grad)

A_ = a.creator
assert A_ is not None
x = A_.input
x.grad = A_.backward(a.grad)
print(x.grad)

A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)

y.grad = np.array(1.0)
y.backward()
print(x.grad)
