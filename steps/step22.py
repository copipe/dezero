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


def as_variable(obj: Union[Variable, np.ndarray, numbers.Number]) -> Variable:
    """Convert np.ndarray to Variable

    Args:
        obj (Union[Variable, np.ndarray, numbers.Number]): Tensor to change Variable

    Returns:
        Variable: Variable Tensor
    """
    if isinstance(obj, Variable):
        return obj
    if isinstance(obj, np.ndarray):
        return Variable(obj)
    return Variable(as_array(obj))


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

    __array_priority__ = 200  # Prefer Variable operators over numpy operators. (ex: np.array([1.0]) + variable(np.array([1.0])))

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

    def __add__(self, other: Union[Variable, np.ndarray, numbers.Number]):
        other = as_variable(other)
        return add(self, other)

    def __mul__(self, other: Union[Variable, np.ndarray, numbers.Number]):
        other = as_variable(other)
        return mul(self, other)

    def __radd__(self, other: Union[Variable, np.ndarray, numbers.Number]):
        other = as_variable(other)
        return add(self, other)

    def __rmul__(self, other: Union[Variable, np.ndarray, numbers.Number]):
        other = as_variable(other)
        return mul(self, other)

    def __neg__(self):
        return neg(self)

    def __sub__(self, other: Union[Variable, np.ndarray, numbers.Number]):
        other = as_variable(other)
        return sub(self, other)

    def __rsub__(self, other: Union[Variable, np.ndarray, numbers.Number]):
        other = as_variable(other)
        return sub(other, self)

    def __truediv__(self, other: Union[Variable, np.ndarray, numbers.Number]):
        other = as_variable(other)
        return div(self, other)

    def __rtruediv__(self, other: Union[Variable, np.ndarray, numbers.Number]):
        other = as_variable(other)
        return div(other, self)

    def __pow__(self, c: int):
        return pow(self, c)


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


class Neg(Function):
    """Forward and backward propagation of negative operation."""

    def forward(self, *xs: np.ndarray) -> np.ndarray:
        return -xs[0]

    def backward(self, gy: np.ndarray) -> np.ndarray:
        return -gy


class Sub(Function):
    """Forward and backward propagation of subtraction operation."""

    def forward(self, *xs: np.ndarray) -> np.ndarray:
        x0, x1 = xs
        y = x0 - x1
        return y

    def backward(self, gy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return gy, -gy


class Div(Function):
    """Forward and backward propagation of division operation."""

    def forward(self, *xs: np.ndarray) -> np.ndarray:
        x0, x1 = xs
        y = x0 / x1
        return y

    def backward(self, gy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1**2)
        return gx0, gx1


class Pow(Function):
    """Forward and backward propagation of power operation."""

    def __init__(self, c: int):
        self.c = c

    def forward(self, *xs: np.ndarray) -> np.ndarray:
        return xs[0] ** self.c

    def backward(self, gy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x = self.inputs[0].data
        c = self.c
        gx = c * x ** (c - 1) * gy
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


def add(x0: Variable, x1: Variable) -> Variable:
    """Perform a add operation.

    Args:
        x0 (Variable): input variable (left)
        x1 (Variable): input variable (right)

    Returns:
        Variable: output variable (x0 + x1)
    """
    y = Add()(x0, x1)
    assert isinstance(y, Variable)
    return y


def mul(x0: Variable, x1: Variable) -> Variable:
    """Perform a multiply operation.

    Args:
        x0 (Variable): input variable (left)
        x1 (Variable): input variable (right)

    Returns:
        Variable: output variable (x0 * x1)
    """
    y = Mul()(x0, x1)
    assert isinstance(y, Variable)
    return y


def neg(x: Variable) -> Variable:
    """Perform a negative operation.

    Args:
        x (Variable): input variable

    Returns:
        Variable: output variable (-x)
    """
    y = Neg()(x)
    assert isinstance(y, Variable)
    return y


def sub(x0: Variable, x1: Variable) -> Variable:
    """Perform a subtraction operation.

    Args:
        x0 (Variable): input variable (left)
        x1 (Variable): input variable (right)

    Returns:
        Variable: output variable (x0 - x1)
    """
    y = Sub()(x0, x1)
    assert isinstance(y, Variable)
    return y


def div(x0: Variable, x1: Variable) -> Variable:
    """Perform a division operation.

    Args:
        x0 (Variable): input variable (left)
        x1 (Variable): input variable (right)

    Returns:
        Variable: output variable (x0 / x1)
    """
    y = Div()(x0, x1)
    assert isinstance(y, Variable)
    return y


def pow(x: Variable, c: int) -> Variable:
    """Perform a power operation.

    Args:
        x (Variable): input variable
        c(int): power number

    Returns :
        Variable: output variable (x^c)
    """
    y = Pow(c)(x)
    assert isinstance(y, Variable)
    return y


x = Variable(np.array(2.0))
y = -x
print(y)

x = Variable(np.array(2.0))
y1 = 2.0 - x  # type:ignore
y2 = x - 1.0  # type:ignore
print(y1)
print(y2)

x = Variable(np.array(2.0))
y1 = 2.0 / x  # type:ignore
y2 = x / 1.0  # type:ignore
print(y1)
print(y2)

x = Variable(np.array(2.0))
y = x**3
print(y)
