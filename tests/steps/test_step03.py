import numpy as np

from steps.step03 import Exp, Square, Variable


def test_composite_function():
    A = Square()
    B = Exp()
    C = Square()

    # y = exp(x^2)^2
    x = Variable(np.array(0.5))
    a = A(x)
    b = B(a)
    y = C(b)

    answer = np.exp(0.5**2) ** 2
    assert answer == y.data
