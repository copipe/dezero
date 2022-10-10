from __future__ import annotations

from typing import Tuple

import numpy as np

from dezero.core import Function, Variable, as_array, as_variable


class Square(Function):
    """Forward and backward propagation of square operation."""

    def forward(self, *xs: np.ndarray) -> np.ndarray:
        return xs[0] ** 2

    def backward(self, gy: np.ndarray) -> np.ndarray:
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx


class Exp(Function):
    """Forward and backward propagation of expornential operation."""

    def forward(self, *xs: np.ndarray) -> np.ndarray:
        return np.exp(xs[0])

    def backward(self, gy: np.ndarray) -> np.ndarray:
        x = self.inputs[0].data
        gx = np.exp(x) * gy
        return gx


class Sin(Function):
    """Forward and backward propagation of sin function."""

    def forward(self, *xs: np.ndarray) -> np.ndarray:
        y = np.sin(xs[0])
        return as_array(y)

    def backward(self, gy: Variable) -> Variable:
        (x,) = self.inputs
        gx = gy * cos(x)
        return gx


class Cos(Function):
    """Forward and backward propagation of cos function."""

    def forward(self, *xs: np.ndarray) -> np.ndarray:
        y = np.cos(xs[0])
        return as_array(y)

    def backward(self, gy: Variable) -> Variable:
        (x,) = self.inputs
        gx = gy * -sin(x)
        return gx


class Tanh(Function):
    """Forward and backward propagation of tanh function."""

    def forward(self, *xs: np.ndarray) -> np.ndarray:
        y = np.tanh(xs[0])
        return as_array(y)

    def backward(self, gy: Variable) -> Variable:
        y = self.outputs[0]()  # outputs[0] is weakref
        gx = gy * (1 - y**2)
        return gx


class Reshape(Function):
    """Forward and backward propagation of reshape function.

    Attributes:
            shape (Tuple): shape of the tensor after reshaping
            x_shape (Tuple): shape of the tensor before reshaping
    """

    def __init__(self, shape: Tuple):
        self.shape = shape

    def forward(self, *xs: np.ndarray) -> np.ndarray:
        x = xs[0]
        self.x_shape = x.shape
        y = x.reshape(self.shape)
        return as_array(y)

    def backward(self, gy: Variable) -> Variable:
        return reshape(gy, self.x_shape)


def square(x: Variable) -> Variable:
    """Perform a square operation.

    Args:
        x (Variable): input variable

    Returns:
        Variable: output variable (x^2)
    """
    y = Square()(x)
    assert isinstance(y, Variable)
    return y


def exp(x: Variable) -> Variable:
    """Perform a expornential operation.

    Args:
        x (Variable): input variable

    Returns:
        Variable: output variable (exp(x))
    """
    y = Exp()(x)
    assert isinstance(y, Variable)
    return y


def sin(x: Variable) -> Variable:
    """Perform a sin function

    Args:
        x (Variable): input variable

    Returns:
        Variable: output variable (sin(x))
    """
    return Sin()(x)


def cos(x: Variable) -> Variable:
    """Perform a cos function

    Args:
        x (Variable): input variable

    Returns:
        Variable: output variable (cos(x))
    """
    return Cos()(x)


def tanh(x: Variable) -> Variable:
    """Perform a tanh function

    Args:
        x (Variable): input variable

    Returns:
        Variable: output variable (tanh(x))
    """
    return Tanh()(x)


def reshape(x: Variable, shape: Tuple) -> Variable:
    """Perform a reshape function

    Args:
        x (Variable): input variable
        shape (Tuple): tensor of shape after reshaping

    Returns:
        Variable: output variable (x.reshape(shape))
    """
    if x.shape == shape:
        return as_variable(x)
    return Reshape(shape)(x)
