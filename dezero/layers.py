from __future__ import annotations

import weakref
from abc import ABCMeta, abstractmethod
from typing import Tuple

import numpy as np

import dezero.functions as F
from dezero.core import Parameter, Variable


class Layer(metaclass=ABCMeta):
    def __init__(self):
        self._params = set()

    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            self._params.add(name)
        super().__setattr__(name, value)

    def __call__(self, *inputs: Variable) -> Variable:
        outputs = self.forward(*inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        self.inputs = [weakref.ref(x) for x in inputs]
        self.outputs = [weakref.ref(y) for y in outputs]
        return outputs if len(outputs) > 1 else outputs[0]

    @abstractmethod
    def forward(self, *inputs: Variable) -> Variable | Tuple[Variable, ...]:
        pass

    def params(self):
        for name in self._params:
            yield self.__dict__[name]

    def cleargrads(self):
        for param in self.params():
            param.cleargrad()


class Linear(Layer):
    def __init__(self, in_size, out_size, nobias=False, dtype=np.float32):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.dtype = dtype

        I, O_ = self.in_size, self.out_size
        W_data = np.random.randn(I, O_).astype(self.dtype) * np.sqrt(1 / I)
        self.W = Parameter(W_data, name="W")

        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_size, dtype=dtype), name="b")

    def forward(self, x: Variable) -> Variable:
        y = F.linear(x, self.W, self.b)
        return y
