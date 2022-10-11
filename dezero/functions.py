from __future__ import annotations

from typing import List, Tuple

import numpy as np

import dezero
import dezero.utils as utils
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


class Reshape(Function):
    """Forward and backward propagation of reshape function.

    Attributes:
            shape (Tuple): shape of the tensor after reshaping
            x_shape (Tuple): shape of the tensor before reshaping
    """

    def __init__(self, shape: Tuple | List):
        self.shape = shape

    def forward(self, *xs: np.ndarray) -> np.ndarray:
        x = xs[0]
        self.x_shape = x.shape
        y = x.reshape(self.shape)
        return as_array(y)

    def backward(self, gy: Variable) -> Variable:
        return reshape(gy, self.x_shape)


class Transpose(Function):
    """Forward and backward propagation of transpose function."""

    def forward(self, *xs: np.ndarray) -> np.ndarray:
        y = np.transpose(xs[0])
        return as_array(y)

    def backward(self, gy: Variable) -> Variable:
        gx = transpose(gy)
        return gx


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


def transpose(x: Variable) -> Variable:
    """Perform a transpose function

    Args:
        x (Variable): input variable

    Returns:
        Variable: output variable (x.T)
    """
    return Transpose()(x)


class Sum(Function):
    """Forward and backward propagation of sum operation.

    Attributes:
        axis (Tuple[int, ...] | int | None, optional): Axis or axes along which a sum is performed.
        keepdims (bool, optional):If this is set to True, the axes which are reduced are left in the result as dimensions with size one. Defaults to False.
        x_shape (Tuple): shape of the tensor before reshaping
    """

    def __init__(
        self, axis: Tuple[int, ...] | int | None = None, keepdims: bool = False
    ):
        self.axis = axis
        self.keepdims = keepdims
        self.x_shape = None

    def forward(self, *xs: np.ndarray) -> np.ndarray:
        x = xs[0]
        self.x_shape = x.shape
        y = x.sum(axis=self.axis, keepdims=self.keepdims)
        return as_array(y)

    def backward(self, gy: Variable) -> Tuple[Variable, Variable]:
        gx = utils.reshape_sum_backward(gy, self.x_shape, self.axis, self.keepdims)
        gx = broadcast_to(gy, self.x_shape)
        return gx


def sum(
    x: Variable, axis: Tuple[int, ...] | int | None = None, keepdims: bool = False
) -> Variable:
    """

    Args:
        x (Variable): input variable
        axis (Tuple[int, ...] | int | None, optional): Axis or axes along which a sum is performed.
        keepdims (bool, optional):If this is set to True, the axes which are reduced are left in the result as dimensions with size one. Defaults to False.

    Returns:
        Variable: output variable (sum(x))
    """
    return dezero.functions.Sum(axis, keepdims)(x)


class BroadcastTo(Function):
    """Forward and backward propagation of broadcast operation.

    Attributes:
        x_shape (Tuple): shape of the tensor before reshaping
        shape (Tuple): shape of the tensor after reshaping
    """

    def __init__(self, shape: Tuple[int, ...]):
        self.shape = shape

    def forward(self, *xs: np.ndarray) -> np.ndarray:
        x = xs[0]
        self.x_shape = x.shape
        y = np.broadcast_to(x, self.shape)
        return as_array(y)

    def backward(self, gy: Variable) -> Variable:
        gx = dezero.utils.sum_to(gy, self.x_shape)
        return gx


class SumTo(Function):
    """Forward and backward propagation of sumto operation.
    This operation is opposite operation of broadcast.

    Attributes:
        x_shape (Tuple): shape of the tensor before reshaping
        shape (Tuple): shape of the tensor after reshaping
    """

    def __init__(self, shape: Tuple[int, ...]):
        self.shape = shape

    def forward(self, *xs: np.ndarray) -> np.ndarray:
        x = xs[0]
        self.x_shape = x.shape
        y = dezero.utils.sum_to(x, self.shape)
        return as_array(y)

    def backward(self, gy: Variable) -> Variable:
        gx = broadcast_to(gy, self.x_shape)
        return gx


class MatMul(Function):
    """Forward and backward propagation of matrix multiply operation."""

    def forward(self, *xs: np.ndarray) -> np.ndarray:
        x0, x1 = xs
        y = x0.dot(x1)
        return as_array(y)

    def backward(self, gy: Variable) -> Tuple[Variable, Variable]:
        x0, x1 = self.inputs
        gx0 = matmul(gy, x1.T)
        gx1 = matmul(x0.T, gy)
        return gx0, gx1


def matmul(x0: Variable, x1: Variable) -> Variable:
    """Perform a matrix multiply operation.

    Args:
        x0 (Variable): input variable (left)
        x1 (Variable): input variable (right)

    Returns:
        Variable: output variable (x0.dot(x1))
    """
    return MatMul()(x0, x1)


def broadcast_to(x: Variable, shape: Tuple[int, ...]):
    """Perform a broadcast operation.

    Args:
        x (Variable): input variable
        shape (Tuple): shape of the tensor after broadcasting

    Returns:
        Variable: output variable after broadcasting
    """
    if x.shape == shape:
        return as_variable(x)
    return BroadcastTo(shape)(x)


def sum_to(x: Variable, shape: Tuple[int, ...]):
    """Perform a sumto operation.
    This operation is opposite operation of broadcast.

    Args:
        x (Variable): input variable
        shape (Tuple): shape of the tensor after sumto

    Returns:
        Variable: output variable after sumto
    """
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)
