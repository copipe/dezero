import numpy as np

from steps.step02 import Square, Variable


def test_square_function():
    x = Variable(np.array(10))
    f = Square()
    y = f(x)
    assert isinstance(y, Variable)
    assert y.data == 100
