from typing import List

import dezero.functions as F
import dezero.layers as L
from dezero import Layer, Variable, utils


class Model(Layer):
    def plot(self, *inputs: Variable, to_file: str = "model.png"):
        y = self.forward(*inputs)
        return utils.plot_dot_graph(y, verbose=True, to_file=to_file)


class MLP(Model):
    def __init__(
        self, input_size: int, fc_output_sizes: List[int], activation=F.sigmoid
    ):
        super().__init__()
        self.activation = activation
        self.layers = []

        sizes = [input_size] + fc_output_sizes
        for i in range(len(sizes) - 1):
            in_size, out_size = sizes[i], sizes[i + 1]
            layer = L.Linear(in_size, out_size)
            setattr(self, "l" + str(i), layer)
            self.layers.append(layer)

    def forward(self, x: Variable) -> Variable:
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)
