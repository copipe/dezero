from __future__ import annotations

import numpy as np

from dezero.core import Function, Variable, as_array


class Sin(Function):
    """Forward and backward propagation of sin function."""

    def forward(self, *xs: np.ndarray) -> np.ndarray:
        y = np.sin(xs[0])
        return as_array(y)

    def backward(self, gy: Variable) -> Variable:
        (x,) = self.inputs
        gx = gy * cos(x)
        return gx


def sin(x: Variable) -> Variable:
    """Perform a sin function

    Args:
        x (Variable): input variable

    Returns:
        Variable: output variable (sin(x))
    """
    return Sin()(x)


class Cos(Function):
    """Forward and backward propagation of cos function."""

    def forward(self, *xs: np.ndarray) -> np.ndarray:
        y = np.cos(xs[0])
        return as_array(y)

    def backward(self, gy: Variable) -> Variable:
        (x,) = self.inputs
        gx = gy * -sin(x)
        return gx


def cos(x: Variable) -> Variable:
    """Perform a cos function

    Args:
        x (Variable): input variable

    Returns:
        Variable: output variable (cos(x))
    """
    return Cos()(x)
