from __future__ import annotations

import numbers
from abc import ABCMeta, abstractmethod
from typing import List, Optional, Tuple, Union

import numpy as np


def as_array(x: Union[np.ndarray, numbers.Number]) -> np.ndarray:
    """Convert numbers.Number to np.ndarray.

    Args:
        x (Union[np.ndarray, numbers.Number]): input tensor(np.ndarray) or number(np.float32, np.float64 etc...).

    Returns:
        np.ndarray: ndarray Tensor

    Notes:
        Adding an operation to 0-dimensional np.ndarray may result in np.float64.
        So if you assume np.ndarray type, you should apply this function in advance.
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
        inputs (Tuple[Variable]): input during forward propagation
        outputs (List[Variable]): input during forward propagation
    """

    def __call__(self, *inputs: Variable) -> Union[Variable, List[Variable]]:
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = outputs
        return outputs if len(outputs) > 1 else outputs[0]

    @abstractmethod
    def forward(self, *xs: np.ndarray) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
        pass

    @abstractmethod
    def backward(self, gy: np.ndarray) -> np.ndarray:
        pass


class Add(Function):
    """Forward and backward propagation of add operation."""

    def forward(self, *xs: np.ndarray) -> np.ndarray:
        x0, x1 = xs
        y = x0 + x1
        return y

    def backward(self, gy: np.ndarray) -> np.ndarray:
        raise NotImplementedError()


def add(x1: Variable, x2: Variable) -> Variable:
    """Perform a add operation.

    Args:
        x1 (Variable): input variable (left)
        x2 (Variable): input variable (right)

    Returns:
        Variable: output variable (x1 + x2)
    """
    y = Add()(x1, x2)
    assert isinstance(y, Variable)
    return y


x1 = Variable(np.array(2.0))
x2 = Variable(np.array(3.0))
y = add(x1, x2)
print(y.data)
