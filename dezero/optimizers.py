from __future__ import annotations

from abc import ABCMeta, abstractclassmethod

import numpy as np

from dezero import Variable
from dezero.layers import Layer
from dezero.models import Model


class Optimizer(metaclass=ABCMeta):
    def __init__(self):
        self.target = None
        self.hooks = []

    def setup(self, target: Model | Layer) -> Optimizer:
        self.target = target
        return self

    def update(self):
        params = [p for p in self.target.params() if p.grad is not None]

        for f in self.hooks:
            f(params)

        for param in params:
            self.update_one(param)

    @abstractclassmethod
    def update_one(self, param):
        pass


def add_hook(self, f):
    self.hooks.append(f)


class SGD(Optimizer):
    def __init__(self, lr: int | float = 0.01):
        super().__init__()
        self.lr = lr

    def update_one(self, param: Variable):
        param.data -= self.lr * param.grad.data


class MomentumSGD(Optimizer):
    def __init__(self, lr=0.01, momentum=0.9):
        super().__init__()
        self.lr = lr
        self.momentum = momentum
        self.vs = {}

    def update_one(self, param):
        v_key = id(param)
        if v_key not in self.vs:
            self.vs[v_key] = np.zeros_like(param.data)

        v = self.vs[v_key]
        v *= self.momentum
        v -= self.lr * param.grad.data
        param.data += v
