import numpy as np


class Variable:
    """A kind of Tensor that is to be considered a module parameter.

    Args:
        data (Tensor): parameter tensor.
    """

    def __init__(self, data: np.ndarray):
        self.data = data


data = np.array(1.0)
x = Variable(data)
print(x.data)
