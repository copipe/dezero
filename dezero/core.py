from __future__ import annotations

import contextlib
import weakref
from abc import ABCMeta, abstractmethod
from numbers import Number
from typing import Any, List, Tuple

import numpy as np

import dezero


class Config:
    enable_backprop: bool = True


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


def as_array(x: np.ndarray | Number | float | int) -> np.ndarray:
    """Convert number to np.ndarray.

    Args:
        x (np.ndarray | Number | float | int): input tensor(np.ndarray) or number(np.float32, float, int etc...).

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
    if np.isscalar(x):
        return np.array(x)
    return x


def as_variable(obj: Variable | np.ndarray) -> Variable:
    """Convert np.ndarray to Variable

    Args:
        obj (Variable | np.ndarray): Tensor to change Variable

    Returns:
        Variable: Variable Tensor
    """
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)


class Variable:
    """A kind of Tensor that is to be considered a module parameter.

    Attributes:
        data (np.ndarray) : parameter tensor.
        name (str | None) : name of this variable.
        grad (float | None): gradient computed by backpropagation.
        creator (Function | None): The function that created this variable.
        generation (int): Generation timing during forward propagation
                          (which serves as a guideline for the processing order during backward propagation)
    """

    __array_priority__ = 200  # Prefer Variable operators over numpy operators. (ex: np.array([1.0]) + variable(np.array([1.0])))

    def __init__(self, data: np.ndarray, name: str | None = None):
        if not isinstance(data, np.ndarray):
            raise TypeError(f"{type(data)} is not supported.")

        self.data = data
        self.name = name
        self.grad: Variable | None = None
        self.creator: Function | None = None
        self.generation: int = 0

    def set_creator(self, func: Function):
        self.creator: Function = func
        self.generation: int = func.generation + 1

    def backward(self, retain_grad: bool = False, create_graph: bool = False):
        if self.grad is None:
            self.grad = Variable(np.ones_like(self.data))

        funcs: List[Function] = []
        seen_set = set()

        def add_func(f: Function) -> None:
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)

        while funcs:
            f = funcs.pop()
            gys = [output().grad for output in f.outputs]  # output is weakref

            with using_config("enable_backprop", create_graph):
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

    def __add__(self, other: Variable | np.ndarray | float | int) -> Variable:
        other = as_variable(as_array(other))
        return add(self, other)

    def __mul__(self, other: Variable | np.ndarray | float | int) -> Variable:
        other = as_variable(as_array(other))
        return mul(self, other)

    def __radd__(self, other: Variable | np.ndarray | float | int) -> Variable:
        other = as_variable(as_array(other))
        return add(other, self)

    def __rmul__(self, other: Variable | np.ndarray | float | int) -> Variable:
        other = as_variable(as_array(other))
        return mul(other, self)

    def __neg__(self) -> Variable:
        return neg(self)

    def __sub__(self, other: Variable | np.ndarray | float | int) -> Variable:
        other = as_variable(as_array(other))
        return sub(self, other)

    def __rsub__(self, other: Variable | np.ndarray | float | int) -> Variable:
        other = as_variable(as_array(other))
        return sub(other, self)

    def __truediv__(self, other: Variable | np.ndarray | float | int) -> Variable:
        other = as_variable(as_array(other))
        return div(self, other)

    def __rtruediv__(self, other: Variable | np.ndarray | float | int) -> Variable:
        other = as_variable(as_array(other))
        return div(other, self)

    def __pow__(self, c: int | float) -> Variable:
        return pow(self, c)

    def reshape(self, *shape: Tuple | List):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return reshape(self, shape)

    @property
    def T(self):
        return transpose(self)

    def sum(self, axis: Tuple[int, ...] | int | None = None, keepdims: bool = False):
        return sum(self, axis, keepdims)


class Function(metaclass=ABCMeta):
    """Base class to create custom Function
    `__call__` applies the operation defined by `forward` to input x (Variable)

    Attributes:
        inputs (Tuple[Variable]): inputs of forward propagation
        outputs (List[Variable]): outputs of forward propagation
        generation (int): Generation timing during forward propagation
                          (which serves as a guideline for the processing order during backward propagation)
    """

    def __call__(self, *inputs: Variable) -> Variable | List[Variable]:
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(y) for y in ys]

        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]
        return outputs if len(outputs) > 1 else outputs[0]

    @abstractmethod
    def forward(self, *xs: np.ndarray) -> np.ndarray | Tuple[np.ndarray, ...]:
        pass

    @abstractmethod
    def backward(self, gy: Variable) -> Variable | Tuple[Variable, ...]:
        pass


class Add(Function):
    """Forward and backward propagation of add operation."""

    def forward(self, *xs: np.ndarray) -> np.ndarray:
        x0, x1 = xs
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 + x1
        return as_array(y)

    def backward(self, gy: Variable) -> Tuple[Variable, Variable]:
        gx0, gx1 = gy, gy
        if self.x0_shape != self.x1_shape:
            gx0 = sum_to(gx0, self.x0_shape)
            gx1 = sum_to(gx1, self.x1_shape)
        return gx0, gx1


class Mul(Function):
    """Forward and backward propagation of multiply operation."""

    def forward(self, *xs: np.ndarray) -> np.ndarray:
        x0, x1 = xs
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 * x1
        return as_array(y)

    def backward(self, gy: Variable) -> Tuple[Variable, Variable]:
        x0, x1 = self.inputs
        gx0, gx1 = gy * x1, gy * x0
        if self.x0_shape != self.x1_shape:
            gx0 = sum_to(gx0, self.x0_shape)
            gx1 = sum_to(gx1, self.x1_shape)
        return gx0, gx1


class Neg(Function):
    """Forward and backward propagation of negative operation."""

    def forward(self, *xs: np.ndarray) -> np.ndarray:
        y = -xs[0]
        return as_array(y)

    def backward(self, gy: Variable) -> Variable:
        return -gy


class Sub(Function):
    """Forward and backward propagation of subtraction operation."""

    def forward(self, *xs: np.ndarray) -> np.ndarray:
        x0, x1 = xs
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 - x1
        return as_array(y)

    def backward(self, gy: Variable) -> Tuple[Variable, Variable]:
        gx0, gx1 = gy, -gy
        if self.x0_shape != self.x1_shape:
            gx0 = sum_to(gx0, self.x0_shape)
            gx1 = sum_to(gx1, self.x1_shape)
        return gx0, gx1


class Div(Function):
    """Forward and backward propagation of division operation."""

    def forward(self, *xs: np.ndarray) -> np.ndarray:
        x0, x1 = xs
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 / x1
        return as_array(y)

    def backward(self, gy: Variable) -> Tuple[Variable, Variable]:
        x0, x1 = self.inputs
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1**2)
        if self.x0_shape != self.x1_shape:
            gx0 = sum_to(gx0, self.x0_shape)
            gx1 = sum_to(gx1, self.x1_shape)
        return gx0, gx1


class Pow(Function):
    """Forward and backward propagation of power operation."""

    def __init__(self, c: int | float):
        self.c = c

    def forward(self, *xs: np.ndarray) -> np.ndarray:
        y = xs[0] ** self.c
        return as_array(y)

    def backward(self, gy: Variable) -> Variable:
        (x,) = self.inputs
        gx = self.c * x ** (self.c - 1) * gy
        return gx


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


class Sum(Function):
    """Forward and backward propagation of sum operation."""

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
        gx = dezero.utils.reshape_sum_backward(
            gy, self.x_shape, self.axis, self.keepdims
        )
        gx = broadcast_to(gy, self.x_shape)
        return gx


class BroadcastTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, *xs: np.ndarray) -> np.ndarray:
        x = xs[0]
        self.x_shape = x.shape
        y = np.broadcast_to(x, self.shape)
        return as_array(y)

    def backward(self, gy: Variable) -> Variable:
        gx = dezero.utils.sum_to(gy, self.x_shape)
        return gx


def broadcast_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return BroadcastTo(shape)(x)


class SumTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, *xs: np.ndarray) -> np.ndarray:
        x = xs[0]
        self.x_shape = x.shape
        y = dezero.utils.sum_to(x, self.shape)
        return as_array(y)

    def backward(self, gy: Variable) -> Variable:
        gx = broadcast_to(gy, self.x_shape)
        return gx


def sum_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)


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
    return Sum(axis, keepdims)(x)
