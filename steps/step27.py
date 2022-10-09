import numpy as np

from dezero import Function, Variable


class Sin(Function):
    """Forward and backward propagation of sin operation."""

    def forward(self, *xs: np.ndarray) -> np.ndarray:
        return np.sin(xs[0])

    def backward(self, gy: np.ndarray) -> np.ndarray:
        x = self.inputs[0].data
        gx = gy * np.cos(x)
        return gx


def sin(x: Variable) -> Variable:
    """Perform a sin operation.

    Args:
        x (Variable): input variable

    Returns:
        Variable: output variable (sin(x))
    """
    y = Sin()(x)
    assert isinstance(y, Variable)
    return y


x = Variable(np.array(np.pi / 4))
y = sin(x)
y.backward()

print(y.data)
print(x.grad)
