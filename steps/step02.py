from abc import ABCMeta, abstractmethod
from re import S

import numpy as np


class Variable:
    def __init__(self, data: np.ndarray):
        self.data = data


class Function(metaclass=ABCMeta):
    def __call__(self, input: Variable):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        return output

    @abstractmethod
    def forward(self, x: np.ndarray):
        pass


class Square(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return x**2


x = Variable(np.array(10))
f = Square()
y = f(x)
print(type(y))
print(y.data)
