from __future__ import annotations

import contextlib
import numbers
import weakref
from abc import ABCMeta, abstractmethod
from typing import Any, List, Optional, Tuple, Union

import numpy as np


class Config:
    enable_backprop = True


@contextlib.contextmanager
def using_config(name: str, value: Any):
    """Temporarily change Config setting.

    Args:
        name (str): value name to change.
        value (Any): value after change.
    """
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)


def no_grad():
    """Temporarily stop backpropagation"""

    return using_config("enable_backprop", False)


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
        name (Optional[str]) : name of this variable.
        grad (float): gradient computed by backpropagation.
        creator (Function): The function that created this variable.
        generation (int): Generation timing during forward propagation
                          (which serves as a guideline for the processing order during backward propagation)
    """

    def __init__(self, data: np.ndarray, name: Optional[str] = None):
        if not isinstance(data, np.ndarray):
            raise TypeError(f"{type(data)} is not supported.")

        self.data = data
        self.name = name
        self.grad: Optional[np.ndarray] = None
        self.creator: Optional[Function] = None
        self.generation: int = 0

    def set_creator(self, func: Function):
        self.creator = func
        self.generation = func.generation + 1

    def backward(self, retain_grad=False):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)

        while funcs:
            f = funcs.pop()
            gys = [output().grad for output in f.outputs]  # output is weakref
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx

                if x.creator is not None:
                    add_func(x.creator)

            if not retain_grad:
                for y in f.outputs:
                    y().grad = None  # y is weakref

    def cleargrad(self):
        self.grad = None

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        p = str(self.data).replace("\n", "\n" + " " * 9)
        return f"variable({p})"

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype


class Function(metaclass=ABCMeta):
    """Base class to create custom Function
    `__call__` applies the operation defined by `forward` to input x (Variable)

    Attributes:
        inputs (Tuple[Variable]): inputs of forward propagation
        outputs (List[Variable]): outputs of forward propagation
        generation (int): Generation timing during forward propagation
                          (which serves as a guideline for the processing order during backward propagation)
    """

    def __call__(self, *inputs: Variable) -> Union[Variable, List[Variable]]:
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]
        return outputs if len(outputs) > 1 else outputs[0]

    @abstractmethod
    def forward(self, *xs: np.ndarray) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
        pass

    @abstractmethod
    def backward(self, gy: np.ndarray) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
        pass


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


class Add(Function):
    """Forward and backward propagation of add operation."""

    def forward(self, *xs: np.ndarray) -> np.ndarray:
        x0, x1 = xs
        y = x0 + x1
        return y

    def backward(self, gy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return gy, gy


class Mul(Function):
    """Forward and backward propagation of multiply operation."""

    def forward(self, *xs: np.ndarray) -> np.ndarray:
        x0, x1 = xs
        y = x0 * x1
        return y

    def backward(self, gy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return gy * x1, gy * x0


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


def mul(x1: Variable, x2: Variable) -> Variable:
    """Perform a multiply operation.

    Args:
        x1 (Variable): input variable (left)
        x2 (Variable): input variable (right)

    Returns:
        Variable: output variable (x1 * x2)
    """
    y = Mul()(x1, x2)
    assert isinstance(y, Variable)
    return y


Variable.__add__ = add
Variable.__mul__ = mul

with no_grad():
    x = Variable(np.array(1.0))
    y = square(x)

a = Variable(np.array(3.0))
b = Variable(np.array(2.0))
c = Variable(np.array(1.0))

y = a * b + c
y.backward()

print(y)
print(a.grad)
print(b.grad)
