from __future__ import annotations

import numbers
from abc import ABCMeta, abstractmethod
from typing import Optional, Union

import numpy as np


def as_array(x: Union[np.ndarray, numbers.Number]) -> np.ndarray:
    """Convert numbers.Number to np.ndarray.

    Args:
        x (Union[np.ndarray, numbers.Number]): input tensor(np.ndarray) or number(np.float32, np.float64 etc...).

    Returns:
        np.ndarray: ndarray Tensor

    Notes:
        Adding an operation to 0-dimensional np.ndarray may result in np.float64
        ```
        >>> x = np.array(1.0)
        >>> y = x ** 2
        >>> print(type(x), type(y))
        <class 'numpy.ndarray'> <class 'numpy.float64'>
        ```
    """
    if isinstance(x, numbers.Number):
        return np.array(x)
    return x


class Variable:
    """A kind of Tensor that is to be considered a module parameter.

    Attributes:
        data (np.ndarray) : parameter tensor.
        grad (float): gradient computed by backpropagation.
        creator (Function): The function that created this variable.
    """

    def __init__(self, data: np.ndarray):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f"{type(data)} is not supported.")

        self.data = data
        self.grad: Optional[np.ndarray] = None
        self.creator: Optional[Function] = None

    def set_creator(self, func: Function):
        self.creator = func

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)
            if x.creator is not None:
                funcs.append(x.creator)


class Function(metaclass=ABCMeta):
    """Base class to create custom Function
    `__call__` applies the operation defined by `forward` to input x (Variable)

    Attributes:
        input (Variable): input during forward propagation
    """

    def __call__(self, input: Variable) -> Variable:
        x = input.data
        y = self.forward(x)
        output = Variable(as_array(y))
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

    def check_type(self):
        raise


class Square(Function):
    """Forward and backward propagation of square operation."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x**2

    def backward(self, gy: np.ndarray) -> np.ndarray:
        x = self.input.data
        gx = 2 * x * gy
        return gx


class Exp(Function):
    """Forward and backward propagation of expornential operation."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.exp(x)

    def backward(self, gy: np.ndarray) -> np.ndarray:
        x = self.input.data
        gx = np.exp(x) * gy
        return gx


def square(x: Variable) -> Variable:
    """Perform a square operation.

    Args:
        x (Variable): input variable

    Returns:
        Variable: output variable (x^2)
    """
    f = Square()
    return f(x)


def exp(x: Variable) -> Variable:
    """Perform a expornential operation.

    Args:
        x (Variable): input variable

    Returns:
        Variable: output variable (exp(x))
    """
    f = Exp()
    return f(x)


x = Variable(np.array(0.5))
x = Variable(None)
# x = Variable(0.5)
